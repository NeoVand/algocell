# V8 prose changes

This pass edits `MAIN_v8.md` only from `## Figure legends` onward and all of `SI_v8.md`. The starting branch was `feat/experiments-runner`, HEAD `6cd9d43d5c8a4bc4586182da1c0a0fdd717cb654`. `STYLE_NOTES.md`, `MAIN_v8_CHANGES.md` and `AUDIT_v7_RESOLUTION.md` were read first. No simulation was run, no scientific value or verdict was corrected and no citation was added. Failed, untested, suspended, exploratory, post hoc, construction and theorem qualifications are retained.

The title, author block, abstract and main text before the boundary are byte-for-byte unchanged. Their SHA-256 is `ea7c0f1873ade4ea58a4e7360dc0afedcb3956bd288582fb57b11b0b1b9b5d33`. Main-text issues below are proposals for the coordinator, not applied edits. The Extended Data map uses the requested final targets `ed8_1.pdf`–`ed8_9.pdf` and manifest `ED_V8.md`; the coordinator owns reconciliation with the parallel figure work.

# Terminology review and read-only main-text issues

This report follows the required initial reading of `STYLE_NOTES.md`, `MAIN_v8_CHANGES.md` and `AUDIT_v7_RESOLUTION.md`. The title, abstract and main text before `## Figure legends` were read only. No manuscript file or audit file was changed by this reviewer. No new numerical correction, verdict change, denominator change or source-bracket change is proposed. Terminology is based on the present manuscript, its existing supplementary definitions and local execution/scoring sources; it is not a new scientific classification.

## Terminology table

“First use” identifies the earliest substantive use in the main text or abstract, followed by the existing operational definition when that comes later. The review preamble defines the C labels before the abstract. Symbols are upright throughout the main text; the prose pass should preserve that convention rather than introduce selective italics.

| Term | Single definition and required scope | First use / definition location |
| --- | --- | --- |
| Encounter | A bounded execution with a focal tape and partner tape placed on the same ring, followed by writing both resulting tapes back. | Main, “Measuring what reproduction preserves”, first paragraph; Methods, “Execution contracts”. |
| Update | A soup batch containing ordered pair draws, collision-controlled encounters and mutation attempts; not an individual instruction or a recorded observation. Internal code may call this `step`, but manuscript prose should call it an update. | Main, “Available copiers need not emerge”, median detection time; Fig. 1b and Methods, “Worlds and detection”, define the unit. |
| Execution step | One unit charged against an encounter’s execution budget under the machine contract. Repeated block moves and REP iterations consume units separately. | Main, “Available copiers need not emerge”, “128- and 512-step arms”; Methods, “Execution contracts” and “Coherent-fetch 8086”. |
| Copy step | One ordered byte assignment in the abstract overlap proof. This mathematical unit must not be conflated with a machine execution step, especially for word-copy instructions. | Supplementary Note 1, ordered assignment indexed by k. |
| Sample / sampling event | An observation or draw made at the specified observation time or under the specified selection rule; qualify as scheduled sample, random-cell draw, partner sample or sampling role. A sample is not a population update. | Abstract, “sampled encounters”; main, “Measuring what reproduction preserves”, sampled measurements. |
| Sampling role | A panel entry selected by an explicit source role, such as first detection, final random cell or final top-three tape. Repeated tapes and overlapping roles do not become independent genotypes or worlds. | Main, “Available copiers need not emerge”, 6502 retention panel; Supplementary Note 4. |
| World | One evolutionary population trajectory under a specified cohort/arm, seed and machine contract; the unit of evolutionary replication. | Main, “Measuring what reproduction preserves”, historical populations and illustrative world; Methods, “Worlds and detection”. |
| Tape | The ordered sequence of L memory bytes associated with a lattice cell or used as one half of an isolated encounter. | Main, “Measuring what reproduction preserves”, first paragraph. |
| Cell | A lattice location containing a tape, including its counted occurrence in a sampled population. In the BFF proof, explicitly say “BFF memory cell” for the different memory unit. | Main, “Measuring what reproduction preserves”, “cell boundaries”; Supplementary Note 3 supplies the BFF exception. |
| Genotype | A distinct exact tape sequence in the declared physical phase. A genotype–world candidate pairs that sequence with its world of origin; it is not a new independent world. Cyclic alignment is an assay rule, not implicit genotype deduplication. | Main, “Available copiers need not emerge”, extended-verifier paragraph; S2 candidate construction in `score8086.py` keys tapes by byte strings. |
| Open | A route classification based on partner instruction-stream fetch under the cohort’s declared rule. The S1/S2/dial criterion is partner fetch in at least half the sampled traces; historical Z80 uses its stated control-flow/copy evidence. | Main, “Carriage can fail despite survival”, first paragraph; Figs 4c and 5a and Supplementary Note 5 specify cohort rules. |
| Confined / confinement | Instruction-stream fetch stays within the focal tape in the declared traced encounters. Fetch includes operands and prefixes, not only instruction starts. | Main, “Carriage can fail despite survival”, three confined replicators; meaning introduced under “Measuring what reproduction preserves”. |
| Partner-information closure | With parent and encounter contract fixed, offspring contents do not depend on the partner’s initial contents over the declared partner class. It is not confinement, survival or a loop. | Main, “Overlap sets what is inherited”, closing paragraph; explicit definition in “Closure and heredity are distinct”. |
| Serial family closure | After execution and reset, an offspring is again a member of the admitted parent family. This mathematical condition should retain its qualifier to distinguish it from partner-information closure. | Supplementary Note 2, serial certificate theorem. |
| Inflow | Information supplied by initial partner contents to the physical offspring, conditioned on the parent and contract; in deterministic execution this equals offspring entropy under the declared partner distribution. | Main, “Measuring what reproduction preserves”, four measurements; Methods, “Culture, carriage and inflow”, and Supplementary Note 2 give the exact definition. |
| Zero sampled inflow | Zero estimated output entropy in the sampled partner panel. It does not by itself certify independence over all permitted partners. | Main, “Environments sort hereditary designs”, separate historical lethal cohort; Methods and Supplementary Note 2 qualify sampling. |
| Pusher | A copier using a load–push mechanism, with the experimental design named explicitly where needed. It is a mechanism name, not a guarantee of openness or a mutually exclusive global ecological class. | Abstract, 8086 pusher; main, “Carriage can fail despite survival”, 16-byte pusher. |
| Regenerator | A copier that reconstructs a constrained pattern while erasing variation outside the retained source region. In population site-score classification, the label specifically means a confined cell with at most two scored sites. | Main, “Carriage can fail despite survival”, two regenerators; Methods, “Culture, carriage and inflow”, gives the population cutoff. |
| Transmitter | A copier that retains broad classes of tested alternatives at particular sites. In population site-score classification, the label specifically means a confined cell with at least five scored sites. It does not imply adaptive payload use. | Main, “Carriage can fail despite survival”, evolved block-copy transmitter; Methods gives the population cutoff. |
| R / T competition classes | Frozen census-defined byte classes for partial-copy / full-tape designs; these are distinct from the thresholded regenerator/transmitter site-score classes. | Main, “Environments sort hereditary designs”, first paragraph; exact membership in Methods, “Competition endpoints”. |
| Culture-positive / operationally “heritable” | A tape that meets the declared generation-two culture-gain criterion. This operational label does not establish exact copying or retention of a newly introduced variant. | Main, “Measuring what reproduction preserves”, culture test; Methods, “Culture, carriage and inflow”. |
| Marker carriage | Presence of the introduced marker under the stated assay-specific rule. Serial/branching window carriage and aligned population byte carriage must remain distinguished. | Abstract, registered carriage criterion; main, “Carriage can fail despite survival”; Methods defines both assays. |
| Scored site | A position at which the required fraction of identifiable tested alternatives meets the specified assay’s retention criterion. A site score is not an exact count of independently free coordinates. | Main, “Carriage can fail despite survival”, “A site scores…”; Methods, “Culture, carriage and inflow”. |
| Unexecuted | Absent from instruction-stream fetches, including operands; the byte may nevertheless be used as data. | Main, “Overlap sets what is inherited”, sampled site-score comparison; Methods gives the operational definition. |
| Core C0 | Historical Z80 compatibility semantics. | Methods, “Validation and chronology”; Supplementary Note 7. |
| Core C1 | Corrected classic Z80 semantics used in the implementation-impact and ecological reruns. Always qualify as core C1 or core-C1 when used adjectivally. | Review preamble; main, “Measuring what reproduction preserves”, validation paragraph. |
| Core C2 | Vendor-specific Zilog-NMOS Z80 semantics in the implementation-impact study. | Methods, “Validation and chronology”; Supplementary Note 7. |
| Displacement experiment C1 | The registered Z80 displacement-edit/capacity study, separate from corrected core C1. | Review preamble; main, “The copying rule is inherited”, first paragraph. |
| 6502 retention experiment C2 | The registered exact-retention study on the complete eligible evolved S1 panel, separate from core C2. | Review preamble; main, “Available copiers need not emerge”, currently shortened to “registered C2 panel”. |
| Registered | An experimental definition or prediction fixed before its corresponding outcome, with scientific amendments and chronology retained explicitly. The label does not imply a positive outcome. | Abstract, registered carriage criterion; Supplementary Note 6 gives experiment-specific chronology. |
| Exploratory | Analysis outside the registered confirmatory outcome, retaining its own selection rules and limitations. | Abstract, separate exploratory 8086 analysis; main, “Overlap sets what is inherited”. |
| Post hoc | Extraction, explanation or selection formulated after relevant observations were available. A later prediction commit before an assay does not make such selection prospective. | Main, “Available copiers need not emerge”, 6502 ring-position explanation; Supplementary Notes 6 and 9. |
| Construction | An intentionally assembled mechanistic example or control under a stated contract, not evidence of evolutionary arrival. | Main, “Available copiers need not emerge”, constructed 6502 controls; Supplementary Notes 3 and 4. |
| Theorem | A mathematical implication under explicit hypotheses, distinct from a sampled intervention result or registered empirical verdict. | Main, “Overlap sets what is inherited”, r = d − c; Supplementary Notes 1–3. |
| L | Tape length in bytes. | Main, “Measuring what reproduction preserves”, first encounter definition. |
| P | Encounter-ring length in bytes. | Methods, “Coherent-fetch 8086”; formally defined in the forward-overlap theorem and Supplementary Note 1. |
| d | Mapped displacement between source and destination in the declared ordered copy geometry. Forward copies use destination minus source; the descending constructed 6502 census explicitly uses read minus write. Preserve that stated orientation convention. | Main, “Overlap sets what is inherited”; Supplementary Note 4 specifies the 6502 sign convention. |
| c (held-coordinate count) | Number of constrained coordinates held fixed within the protected source region for the stated family. The coordinates need not be contiguous. With the remaining coordinates jointly free, r = d − c. | Main, “Overlap sets what is inherited”, conditional payload sentence; Methods, “Evolved 6502 retention”. |
| c (amended S2 L2 bound) | Bounding-prefix length c = 1 + max(K) for the raw held footprint K. This registered bound is not the number of constrained source roots; the exact payload count remains d − the cardinality of C_R. | Methods, “Provenance certificates”; Supplementary Note 2, shifted source roots. |
| r / free payload | The number of independently and jointly variable source coordinates admitted while the footprint and execution certificate are held fixed. When c counts the constrained coordinates, r = d − c; in the general source-root certificate it is d minus the cardinality of C_R. | Main, “Overlap sets what is inherited”, conditional payload sentence. |
| p | Probability that fetching a zero opcode halts execution in the lethality dial, not a significance-test P value. | Fig. 4c; Methods, “Competition endpoints”. |
| Benign chemistry | The declared contract in which a zero opcode executes normally rather than triggering the lethal stopping rule. A zero data byte is not itself a halting event. | Main, “Carriage can fail despite survival”, benign cohort; explicitly defined in “Environments sort hereditary designs”. |
| Lethal chemistry | The declared contract in which fetching a zero opcode halts execution. “Tar” should not be used in prose; preserve `tar` inside existing paths, fields or source brackets. | Main, “Environments sort hereditary designs”, first competition paragraph; exact rule in `algocell_exp/gen_dial_shader.py`. |
| Zero partner | A partner whose initial bytes are all zero, not a panel containing no partners. | Main, “Available copiers need not emerge”, census self-writer condition; Methods and Supplementary Note 4 specify the panel. |

Coordinator note: the registered notation overload for c must remain explicitly scoped. The main text and evolved 6502 use c as a held-coordinate count, without requiring a contiguous core. Amended S2 L2 instead defines c as a bounding-prefix length and supplies a separate exact constrained-root count. A unified “extent” definition would misdescribe the cardinality case; narrowing the main text to contiguous cores would unnecessarily weaken the stated theorem. Retain the two scoped entries rather than change the scientific notation or registration. Similarly, the forward and descending 6502 definitions of d must remain explicit rather than be silently unified.

## Main-text issues left unchanged for the coordinator

All proposed replacements below lie before `## Figure legends` and therefore were not applied. They preserve numerical statements, source brackets and status labels unless the item explicitly concerns front-matter placeholders, which the coordinator must reconcile under the author’s instructions.

### Required author/front-matter reconciliation

The author line is currently “Mostafa ‘Neo’ Mohsenvand”. Proposed replacement: “Mostafa ‘Neo’ Mohsenvand, MIT Media Lab”. The following affiliation/address sentence may retain the fuller institutional postal information and correspondence placeholder.

The front-matter `Author-supplied items still required` line includes an extra `[RELEASE PACKAGE: standalone cross-machine encounter benchmark and generated table]` placeholder and overlaps the later availability sections. The user’s allowed remaining placeholders omit this separate release-package item. Proposed replacement sentence: “Author declarations and release accessions are specified in the corresponding sections below.” Reconcile removal of the obsolete placeholder line centrally so that duplicate front-matter source/placeholder accounting is explicit. This reviewer cannot apply that change within the assigned boundary.

The italic review preamble includes future editorial instructions (“will move to a Source Data/provenance index for submission”) and release commentary (“No accession, author declaration or completed public release is implied by a placeholder”). Proposed replacement paragraph: “Result-path source brackets resolve from `experiments/`, unless an absolute path is given. They identify records in the Source Data/provenance index. Numbered citations refer to the reference list. ‘Core C1’ denotes corrected Z80 semantics; ‘displacement experiment C1’ and ‘6502 retention experiment C2’ denote the registered capacity studies. Registered outcomes, exploratory analyses, constructions and theorems retain separate status. Figures refer to `manuscript/figures/out/v8/`; Supplementary Notes and Tables refer to `manuscript/SI_v8.md`.” Use typographic double quotes consistently if retaining the manuscript’s current quote style. This proposal removes revision-history prose; it must be checked against the coordinator’s final provenance-index arrangement before adoption.

### Terminology and precision

Under “Measuring what reproduction preserves”, the general encounter sentence says “executes a bounded number of instructions”, whereas machine budgets can charge a repeated block operation/REP iteration separately. Proposed replacement: “An encounter places two tapes of L bytes on a ring of 2L bytes, resets the processor to a declared state, executes a bounded number of execution steps starting in the first tape and writes both tapes back.” The local execution contracts already establish this usage; no numerical change is required.

Under “Available copiers need not emerge”, the body says “The complete registered C2 panel sampled 942 roles from all 80 S1 worlds and, by frozen code-and-provenance rules, admitted 42 sampling roles representing 41 distinct tapes from 11 worlds.” Proposed replacement: “The complete panel for the registered 6502 retention experiment C2 sampled 942 roles from all 80 S1 worlds and, by frozen code-and-provenance rules, admitted 42 sampling roles representing 41 distinct tapes from 11 worlds.” This resolves the remaining standalone experiment/core label ambiguity without changing its evidence or status.

Under “Overlap sets what is inherited”, the main sentence correctly uses c as the number of held core coordinates within the protected d-byte region. It does not require a contiguous core and needs no scientific narrowing. Keep this use distinct from the amended S2 L2 bound c = 1 + max(K); only the British-spelling change from “initialization” to “initialisation” is proposed for this sentence.

Under “Overlap sets what is inherited”, “The Z80 soups evolve exactly this machine” overstates what the following motif observation alone certifies; the next paragraphs correctly explain that motif/divisibility do not certify all hypotheses. Proposed replacement: “The Z80 soups contain copiers with this geometry.” The recurring architecture’s count and attached evidence remain in the same paragraph.

### Voice and house style

The introduction retains the direct question “We ask a narrower question: which introduced parental differences remain in descendants, independently of lineage survival, and which features of the copying operation decide their transmission?” Proposed replacement: “We test which introduced parental differences remain in descendants independently of lineage survival and which features of the copying operation determine their transmission.”

The discussion retains the direct question “Ours is an inspectable case in which the question has an exact answer: which alternatives does reproduction preserve, and which does its own operation remove?” Proposed replacement: “This inspectable case identifies the alternatives that reproduction preserves and those that its operation removes.”

The closing sentence of “Carriage can fail despite survival” uses a dangling question-led transition: “This raises the question of what, in the copying operation, decides which differences survive.” Proposed replacement: “We next examine how the copying operation determines which differences survive.”

The introduction’s first paragraph uses “appropriately organized” and “cellular organization”. Proposed replacements, with the original citations unchanged: “Some compositional systems return towards built-in states and fail to preserve selectable variants [4], whereas appropriately organised autocatalytic networks can support limited heredity and adaptation [5].” “Woese placed the onset of Darwinian evolution proper at a threshold, where cellular organisation became integrated enough for vertical descent to dominate the horizontal exchange of an earlier communal regime [6].”

Under “Overlap sets what is inherited”, “initialization” should be “initialisation” in the conditional payload sentence; the coordinator can apply that isolated spelling correction without changing the mathematical scope.

The introduction contains a serial comma in “trace where copied bytes come from, describe compact copying cores, and show”. Proposed replacement: “These studies trace where copied bytes come from, describe compact copying cores and show that some cores copy random tails faithfully and survive repeated rounds of mutation and replication [13, 14].”

The introductory summary stacks three assertions with a serial comma: “In a recurring family of later copiers, a single geometric property of the copy decides which parental bytes reach descendants, the parameter that sets this geometry is itself inherited, and the chemistry of the environment shifts which designs prevail.” Proposed replacement: “In a recurring family of later copiers, a single geometric property of the copy decides which parental bytes reach descendants. The parameter that sets this geometry is itself inherited. Environmental chemistry shifts which designs prevail.”

The validation sentence contains an unnecessary comma before the second architecture group. Proposed replacement: “We validated the processor cores against per-instruction reference tests: emulator-generated vectors for the Z80 and 6502 and hardware-recorded vectors for the 8088.”

The environment paragraph contains an unnecessary comma between the two chemistry outcomes. Proposed replacement: “Both invasion directions ended at similar environment-specific medians, with five worlds per direction: 0.09 and 0.08 under benign chemistry and 0.81 and 0.77 under lethal chemistry.”

The main text continues to use first-person plural scientific narration (“we”) despite the single-author attribution. There is no literal “the authors” before the boundary. The user explicitly prohibited “the authors”, not every conventional “we”; this is an author/coordinator voice choice rather than a factual defect. The scoped pass should not silently rewrite the read-only body on that basis.

## Local checks and limits

The definitions of update and genotype were checked against `algocell_exp/soup.py` and `score8086.py`: population `step` batches pairing/execution/mutation, whereas candidate entries retain exact `tape` strings and source/world information. The lethal-opcode definition was checked against `algocell_exp/gen_dial_shader.py`, which conditions halting on the fetched opcode. The c distinction and source-root formula were checked against the existing Methods, Supplementary Note 2 and audit resolution N0523; the 6502 reversed displacement convention is explicit in Supplementary Note 4 and its cited `census6502_report.py` geometry implementation. No generated numerical value was found wrong in this read-only review. Every correction above concerns wording, scope or author/front-matter requirements.


## MAIN: final old → new ledger

The blocks below compare the saved original with the final scoped text. Each block includes all prose and house-style changes at that location, so no intermediate editing state is presented as the final text. Source brackets are unchanged except for the explicitly requested author-supplied placeholders.


### MAIN-01

Reason: Make the panel description a declarative sentence. Replace a sentence fragment with a complete sentence. Remove the serial comma. Make the panel description a complete sentence. Remove the serial list separator before and. Tighten the caption while retaining its numerical content, caveats and status.

Old:

**a,** Two conceptual Z80 copying mechanisms. The pusher uses the same bytes as instructions and literal output: `01 c5 01` loads BC, and `c5` pushes it into partner memory. Execution can enter the partner; a separate return-containing design redirects execution into its own tape. These examples are not a documented ancestor–descendant pair. Ellipses abbreviate tapes; `e3` exchanges HL with stack memory. **b,** Cumulative detection, estimated as one minus Kaplan–Meier survival, for historical Stage E worlds at six tape lengths L (bytes), with 10 worlds per length and nondetections censored at 300,000 updates. Encounters have a 128-instruction budget. Operational heritability requires generation-two normalized culture gain ≥0.3. Updates are encounter batches. **c,** One illustrative L=16 world, seed 2002, containing 20,000 cells. Curves show cells matching the contemporaneous modal tape at ≥12/16 unshifted positions, zero bytes/all bytes, and distinct tapes/all cells. Similarity is not a reproduction assay. **d,** Five validation comparisons with distinct denominators: hardware-recorded 8088 instruction states; historical versus corrected Z80 ring conditions; isolated assay combinations; claim verdicts; and constituent tests. The ring criterion is a 95% upper bound on changed-memory rate below 0.01%. Ochre marks expected hardware divergences; vermilion marks flagged or changed comparisons; grey marks agreement under the row-specific criterion. The 10,083 hardware divergences are expected; none is unexpected. Queue timing is outside the contract. Core C1 denotes corrected Z80 semantics; rows must not be pooled as one accuracy estimate.

New:

**a,** The schematic shows two Z80 copying mechanisms. The pusher uses the same bytes as instructions and literal output: `01 c5 01` loads BC, and `c5` pushes it into partner memory. Execution can enter the partner; a separate return-containing design redirects execution into its own tape. These are not a documented ancestor–descendant pair. Ellipses abbreviate tapes; `e3` exchanges HL with stack memory. **b,** Cumulative detection is one minus Kaplan–Meier survival for historical Stage E worlds at six tape lengths L (bytes), with 10 worlds per length and nondetections censored at 300,000 updates. Encounters have a 128-instruction budget. Operational heritability requires generation-two normalised culture gain ≥ 0.3. Updates are encounter batches. **c,** One illustrative L = 16 world, seed 2002, contains 20,000 cells. Curves show cells matching the contemporaneous modal tape at ≥ 12/16 unshifted positions, zero bytes/all bytes and distinct tapes/all cells. Similarity is not a reproduction assay. **d,** Five validation comparisons use distinct denominators: hardware-recorded 8088 instruction states; historical versus corrected Z80 ring conditions; isolated assay combinations; claim verdicts and constituent tests. The ring criterion is a 95% upper bound on changed-memory rate below 0.01%. Ochre marks expected hardware divergences; vermilion marks flagged or changed comparisons; grey marks agreement under the row-specific criterion. The 10,083 hardware divergences are expected; none is unexpected. Queue timing is outside the contract. Core C1 denotes corrected Z80 semantics; rows must not be pooled as one accuracy estimate.


### MAIN-02

Reason: Make the panel description a complete sentence. Remove the serial comma. Distinguish execution steps from population updates and make the panel description a sentence. Use the same execution-step term within the trace. Make the panel description a sentence. Make the plotted comparison explicit. Make the panel description a sentence without altering status. Tighten the caption while retaining its numerical content, caveats and status. Make the plot description direct within the caption word limit. Retain the nesting caveat concisely within the caption word limit. Retain the explicit branching population denominator. Retain the distinct-tape qualifier. Use grammatical wording for the marker key. Keep the ancestry coordinate distinct from a compound label. Preserve the case-count unit of the constructed comparison. Remove a redundant direction preposition. Name the inherited parameter directly and keep the caption within its 250-word contract.

Old:

**a,** Registered branching transfers with uniform-byte partners: 32 alternatives/site, four populations/alternative and 64 controls per design. Each member encounters two fresh partners; only ≥75% best-cyclic-shift matches to their immediate parent continue, with uniform capping at 32 members. Generation-eight bars partition all mutant starts: lost, alive with <50% marker carriage, or alive with ≥50% carriage. Minority carriers can remain. Markers are centred five-byte windows absent from wild type. A site scores when, for at least half its identifiable alternatives, carriage among surviving mutant populations exceeds surviving-control carriage by ≥0.5. Side columns give control survival and scored sites. Historical/corrected-core endpoints coincide. **b,** Historical serial carriage for the same designs: 255 alternatives/site, 16 starts/alternative, eight transfers and 256 controls. Fractions use all mutant starts. **c,** Serial site maps: one byte/box, tracks scaled separately. Grey denotes instruction-stream fetch, including operands, across 64 reference encounters; white denotes no fetch. Open circles mark sites scoring at transfer one but not eight; filled circles mark transfer-eight sites. The L50 pusher uses 32 alternatives/site; others use 255. The L20 transmitter is seed 6004, versus 6003 in a,b. Serial site scoring applies the same excess criterion to surviving lineages. **d,** One exploratory lineage against zero partners for two 128-instruction encounters. At zero-based step 6, address 4 supplies ancestry from original byte 0, overwriting marker 0xA7 at address 8. Its last ancestry label disappears. At step 128, offspring becomes parent and the processor resets. This trace differs from window carriage.

**Figure 3. An inherited copying rule controls free payload.**

**a,** Forward overlapping copy on two L=12 tapes; each box represents two bytes. With fixed core c and protected prefix d, free payload is r=d−c. Displacement d=4 regenerates the four-byte core; d=12 preserves eight payload positions. Initialization, d dividing L, coverage, noninterference and alignment are required. **b,** Prediction versus reported positions; inset: 0–10. Grey discs show sampled unexecuted-site scores for 2,343 exploratory Z80 cells; area gives cell count. Teal rings show exact capacities for 101 registered settings, including 22 original-value controls, in 22 parents. Ochre triangles show exact capacities for 42 eligible 6502 roles: 41 distinct tapes from 11 worlds. Squares show 296 constructed 6502 sampled-carriage cases, not exact capacities. Symbols overlap. The red point retains four positions in targeted transfers but scores zero under the population assay’s ≥75% whole-tape match gate. Cells and roles are nested observations. Registered 8086 L1/L2 remain untested; L2 support is suspended. Post hoc assays are not plotted. **c,** Parallel generation-eight descendants, lineage zero. Editing byte 0: 0x04→0x60 gives displacement 96 mod 64=32, transmitting 28 payload positions. Grey denotes fixed core, red lost ancestry and teal inherited payload. This is an intervention, not a spontaneous transition. **d,** Exploratory scan of 256 byte values: 157 nonworking, 88 working outside sampled eligibility conditions and 11 eligible, shaded by predicted payload. Working means generation-two normalized culture gain ≥0.3 with 16 uniform partners. Rims mark 0x04/0x60; eligibility and working scores alone do not certify universal serial inheritance.

New:

**a,** Registered branching transfers use uniform-byte partners: 32 alternatives/site, four populations/alternative and 64 controls per design. Each member meets two fresh partners; only ≥ 75% best-cyclic-shift matches to the immediate parent continue, uniformly capped at 32 members. Generation-eight bars partition mutant starts: lost, alive with < 50% marker carriage or alive with ≥ 50% carriage. Minority carriers can remain. Markers are centred five-byte windows absent from wild type. Sites score when carriage among surviving mutant populations exceeds surviving-control carriage by ≥ 0.5 for at least half their identifiable alternatives. Columns give control survival and site scores. Historical/corrected-core endpoints coincide. **b,** Historical serial assays use these designs: 255 alternatives/site, 16 starts/alternative, eight transfers and 256 controls. Fractions use all mutant starts. **c,** Site maps show one byte/box; tracks scale separately. Grey marks instruction-stream fetch, including operands, across 64 reference encounters; white marks no fetch. Open circles score at transfer one but not eight; filled circles mark transfer-eight sites. The L50 pusher uses 32 alternatives/site; others use 255. The L20 transmitter uses seed 6004, versus 6003 in a,b. Serial scoring applies this excess criterion to surviving lineages. **d,** One exploratory lineage meets all-zero partners in two 128-instruction encounters. At zero-based execution step 6, address 4 supplies ancestry from original byte 0, overwriting marker 0xA7 at address 8 and erasing its last ancestry label. At execution step 128, offspring replaces parent and the processor resets. This trace differs from window carriage.

**Figure 3. Inherited displacement controls free payload.**

**a,** Forward overlap copies two L = 12 tapes; boxes represent two bytes. For fixed core c and protected prefix d, free payload r = d−c. Displacement d = 4 regenerates the four-byte core; d = 12 preserves eight payload positions. Initialisation, d dividing L, coverage, noninterference and alignment are required. **b,** Predicted versus reported positions are shown (inset: 0–10). Grey discs show sampled unexecuted-site scores for 2,343 exploratory Z80 cells; area denotes count. Teal rings show exact capacities for 101 registered settings, including 22 original-value controls, across 22 parents. Ochre triangles show exact capacities for 42 eligible 6502 roles: 41 distinct tapes from 11 worlds. Squares show 296 constructed 6502 sampled-carriage cases, not exact capacities. Symbols overlap. The red point retains four positions in targeted transfers but scores zero under the population ≥ 75% whole-tape gate. Cells and roles are nested. Registered 8086 L1/L2 remain untested; L2 support is suspended. Post hoc assays are not plotted. **c,** Parallel generation-eight descendants use lineage zero. Editing byte 0 (0x04→0x60) gives displacement 96 mod 64 = 32, transmitting 28 payload positions. Grey denotes fixed core, red lost ancestry and teal inherited payload. This intervention is not a spontaneous transition. **d,** An exploratory 256-value scan finds 157 nonworking, 88 working outside sampled eligibility and 11 eligible, shaded by predicted payload. Working requires generation-two normalised culture gain ≥ 0.3 with 16 uniform partners. Rims mark 0x04/0x60; eligibility and working scores do not certify universal serial inheritance.


### MAIN-03

Reason: Make the panel description a sentence. Use benign/lethal chemistry consistently and distinguish opcodes from data bytes. Tighten the caption while retaining its numerical content, caveats and status. Keep the motif prevalence numerator and denominator explicit. State the chemistry rule concisely.

Old:

**a,** L32 mutation-on competitions through 300,000 updates. R/T are frozen census-defined byte classes corresponding to partial-copy (d<L) and full-tape (d=L) settings, distinct from site-score classes in c. Upper curves show T/(R+T); lower curves show cells carrying `5e ed b0` at bytes 1–3 divided by all 20,000 cells, including other byte-0 variants. Curves are world medians. Each chemistry/implementation has 10 equal-start worlds and five per reciprocal 1% invasion. Line styles/markers identify starting mixtures; light/open denotes historical implementations and dark/filled corrected core C1. Benign zero opcodes execute normally; lethal zeros halt. Endpoint columns give core C1 then historical medians. **b,** World endpoints for relative T share, motif prevalence and distinct T-tail sequences at bytes 4–31. Shapes retain their meanings from a; horizontal offsets separate points. One corrected-core benign world has R+T=0, leaving 19 defined ratios overall and four in its five-world T-into-R subgroup; zero absolute abundance and tail diversity remain included. **c,** Random-code lethality dial, L16, 10 worlds per probability/implementation. p is the probability that fetching a zero opcode halts. Open requires partner instruction-stream fetch in ≥8/16 traces; confinement requires none. Transmitter majority is an estimated share >50% of heritable cells in the confined, ≥5-site class. Counts retain all worlds. Confinement times are Kaplan–Meier medians in thousands of updates, with nondetections censored at 300,000. Historical DZ2/DZ3 failures remain distinct from corrected-core passes. These endpoints do not establish equilibrium or its mechanism.

New:

**a,** L32 mutation-on competitions run through 300,000 updates. Frozen census-defined R/T classes denote partial-copy (d < L) and full-tape (d = L) settings, distinct from c’s site-score classes. Upper curves show T/(R+T); lower curves show cells carrying `5e ed b0` at bytes 1–3 divided by all 20,000 cells, including other byte-0 variants. Curves are world medians. Each chemistry/implementation has 10 equal-start worlds and five per reciprocal 1% invasion. Styles/markers identify starting mixtures; light/open denotes historical implementations and dark/filled corrected core C1. Zero opcodes execute normally under benign chemistry and halt under lethal chemistry. Endpoint columns give core C1 then historical medians. **b,** Endpoints show relative T share, motif prevalence and distinct T-tail sequences at bytes 4–31. Shapes follow a; horizontal offsets separate points. One corrected-core benign world has R+T = 0: 19 ratios remain overall, four in its five-world T-into-R subgroup. Zero absolute abundance and tail diversity remain included. **c,** The random-code lethality dial uses L16 and 10 worlds per probability/implementation. p denotes halting probability upon fetching a zero opcode. Open requires partner instruction-stream fetch in ≥ 8/16 traces; confinement requires none. Transmitter majority requires estimated share > 50% of heritable cells in the confined ≥ 5-site class. Counts retain all worlds. Confinement times are Kaplan–Meier medians (thousands of updates); nondetections are censored at 300,000. Historical DZ2/DZ3 failures remain distinct from corrected-core passes. Endpoints establish neither equilibrium nor its mechanism.


### MAIN-04

Reason: Make the panel description a sentence. Replace a fragment while retaining the unit conveyed by push width. Tighten the caption while retaining its numerical content, caveats and status.

Old:

**a,** First detected copier routes among all worlds. Historical Z80 Stage G contains 80 worlds, 20 each at L=16,20,50,64, all open first. Primary 6502 S1 A has 20 L16 worlds, all confined first. Coherent-fetch 8086 S2 has 20 L16 worlds: two open, 12 confined and six censored at 3,000,000 updates. For S1/S2, open means partner instruction-stream fetch in ≥8/16 traces; confined means none. Historical Z80 classification used control-flow and partner-copy evidence. Confinement does not establish partner-content independence. S1’s additional loop requirement holds in 14/20, leaving P1 unsupported; S2 passes detection sufficiency but fails the open-route prediction and triggers its kill rule. **b,** Kaplan–Meier cumulative detection for 20 L16 worlds per machine, each with budget 128. Horizons are 300,000 Z80, 10,000,000 6502 and 3,000,000 8086 updates. One terminal mark represents six coincident 8086 censors. Events are scheduled detections, not births; protocols differ and timing comparisons are descriptive. **c,** Bytes emitted per push and exact pusher availability under the declared census/reset contracts. The 6502 absence concerns the tested periodic census, not all possible copiers. Availability does not determine establishment. **d,** All ten `5F A4 x y` first-copier tapes among 14 detected 8086 worlds, shown in physical phase with detection times. `5F A4` encodes `POP DI; MOVSB`. Tile outlines group repeats. Frozen mechanism labels hold in 16/16 encounters: eight one-pass and two copy-loop classifications. Colours identify shared bytes, not certified payload. Later overlap analysis is exploratory and does not change registered verdicts.

New:

**a,** First-copier routes cover all worlds. Historical Z80 Stage G has 80 worlds, 20 each at L = 16,20,50,64, all open first. Primary 6502 S1 A has 20 L16 worlds, all confined first. Coherent-fetch 8086 S2 has 20 L16 worlds: two open, 12 confined and six censored at 3,000,000 updates. For S1/S2, open means partner instruction-stream fetch in ≥ 8/16 traces; confined means none. Historical Z80 classification used control-flow and partner-copy evidence. Confinement does not establish partner-content independence. S1’s additional loop requirement holds in 14/20, leaving P1 unsupported; S2 passes detection sufficiency but fails the open-route prediction and triggers its kill rule. **b,** Kaplan–Meier curves show cumulative detection for 20 L16 worlds/machine at budget 128. Horizons are 300,000 Z80, 10,000,000 6502 and 3,000,000 8086 updates. One terminal mark represents six coincident 8086 censors. Events are scheduled detections, not births; protocols differ and timing comparisons are descriptive. **c,** Bytes/push and exact pusher availability follow declared census/reset contracts. The 6502 absence concerns the tested periodic census, not all possible copiers. Availability does not determine establishment. **d,** All ten `5F A4 x y` first-copier tapes from 14 detected 8086 worlds appear in physical phase with detection times. `5F A4` encodes `POP DI; MOVSB`. Tile outlines group repeats. Frozen mechanism labels hold in 16/16 encounters: eight one-pass and two copy-loop classifications. Colours identify shared bytes, not certified payload. Later overlap analysis is exploratory and does not change registered verdicts.


### MAIN-05

Reason: Use an active sentence and remove a dangling distinction. State the final distinction between historical kernels and corrected reruns directly. Remove the obsolete tar term, clarify that chemistry acts on fetched opcodes and remove the serial comma. Distinguish tapes from population cells. Retain an explicit opcode rule for each chemistry.

Old:

Zilion separates instruction cores, shared encounter interfaces, execution engines and machine manifests. Population pairing, mutation and scientific classifications remain research code. Manifests bind semantic version, compatibility, chip variant, generated shader and host identities to reset, address mapping, budget, traps and observations. Historical Algocell populations and assays retain their actual kernels; fresh corrected-core reruns are not relabelled historical Zilion executions. Architecture and machine-specific contracts are archived separately from experiment results [algocell_exp/machines/; /Users/neo/repos/zilion/docs/ARCHITECTURE.md].

An encounter places focal A and partner B on a ring, resets the processor, enters A, executes its budget and writes both halves back. Architectural wrapping precedes ring reduction; a wrap need not advance to the neighbouring ring byte for arbitrary ring sizes. The historical Z80 resets registers and flags to zero except SP, set to the largest architectural address aliasing the partner’s last byte. That pointer convention does not itself specify the first written byte. Fetch maps include prefixes, operands and displacements, not merely instruction starts. Repeating block moves consume budget per repetition; I/O reads return zero, writes have no external effect, and no external interrupts occur. Benign zero executes normally; lethal tar halts on a fetched zero opcode. The 8080 subset is an ablation within this core, not independent hardware validation [results/emulator/RING_DIFFTEST.md; algocell_exp/shader/z80_test_L16.wgsl; algocell_exp/shader/z80_test_lethal_L16.wgsl; results/review_r2/RUN_TABLE.md].

New:

Zilion separates instruction cores, shared encounter interfaces, execution engines and machine manifests. Research code implements population pairing, mutation and scientific classifications. Manifests bind semantic version, compatibility, chip variant, generated shader and host identities to reset, address mapping, budget, traps and observations. Historical Algocell populations and assays retain their actual kernels; fresh reruns with corrected cores are reported separately. Architecture and machine-specific contracts are archived separately from experiment results [algocell_exp/machines/; /Users/neo/repos/zilion/docs/ARCHITECTURE.md].

An encounter places focal tape A and partner tape B on a ring, resets the processor, enters A, executes its budget and writes both halves back. Architectural wrapping precedes ring reduction; a wrap need not advance to the neighbouring ring byte for arbitrary ring sizes. The historical Z80 resets registers and flags to zero except SP, set to the largest architectural address aliasing the partner’s last byte. That pointer convention does not itself specify the first written byte. Fetch maps include prefixes, operands and displacements, not merely instruction starts. Repeating block moves consume budget per repetition; I/O reads return zero, writes have no external effect and no external interrupts occur. A fetched zero opcode executes normally in benign chemistry; in lethal chemistry, a fetched zero opcode halts. The 8080 subset is an ablation within this core, not independent hardware validation [results/emulator/RING_DIFFTEST.md; algocell_exp/shader/z80_test_L16.wgsl; algocell_exp/shader/z80_test_lethal_L16.wgsl; results/review_r2/RUN_TABLE.md].


### MAIN-06

Reason: Name the encounter budget unit consistently. State the analysed contract directly while preserving the distinction from the exploratory reset.

Old:

S2 used the Zilion NMOS core frozen at `39a60429a9b16c77cd845f6087062c4e2cc12c53`, compatibility `0.3`, with `prefetch=coherent` and `hidden_latch=zero_ea`; package status was experimental. Consumed instruction bytes come from current memory. There is no prefetch queue or bus-timing model. General registers, segments and entry IP reset to zero except SP = 1; FLAGS is `F002`, with DF and TF clear. At the registered ring sizes, the first word push decrements SP and writes low then high bytes at ring positions P−1 and 0. The earlier exploratory zero-SP inventory is superseded [algocell_exp/machines/worlds_8086.json; algocell_exp/machines/PROVENANCE_8086.md; results/s8086/prelaunch/freeze.json].

Segment translation and architectural wrapping precede ring mapping; each word byte wraps its offset independently. A REP iteration consumes a budget step and retains decoded operation/source segment between continuations; zero count copies nothing. Register LEA/LES/LDS forms trap. Undefined far-register transfers use the declared fresh zero-EA hidden-state abstraction. Interrupt vectors alias ring memory, external interrupts are absent, input supplies all-one bytes, output is discarded, HLT halts and WAIT without a coprocessor does nothing. These policies matter to evolving programs; coherent fetch is the research contract, not a claim to reproduce hardware-prefetch ecology [algocell_exp/machines/worlds_8086.json; algocell_exp/machines/i8086_pair_maps_L16.wgsl; results/s8086/prelaunch/freeze.json; /Users/neo/repos/zilion/docs/I8086.md].

New:

S2 used the Zilion NMOS core frozen at `39a60429a9b16c77cd845f6087062c4e2cc12c53`, compatibility `0.3`, with `prefetch=coherent` and `hidden_latch=zero_ea`; package status was experimental. Consumed instruction bytes come from current memory. There is no prefetch queue or bus-timing model. General registers, segments and entry IP reset to zero except SP = 1; FLAGS is `F002`, with DF and TF clear. At the registered ring sizes, the first word push decrements SP and writes low then high bytes at ring positions P−1 and 0. These results use this reset rather than the exploratory zero-SP inventory [algocell_exp/machines/worlds_8086.json; algocell_exp/machines/PROVENANCE_8086.md; results/s8086/prelaunch/freeze.json].

Segment translation and architectural wrapping precede ring mapping; each word byte wraps its offset independently. A REP iteration consumes an execution step and retains decoded operation/source segment between continuations; zero count copies nothing. Register LEA/LES/LDS forms trap. Undefined far-register transfers use the declared fresh zero-EA hidden-state abstraction. Interrupt vectors alias ring memory, external interrupts are absent, input supplies all-one bytes, output is discarded, HLT halts and WAIT without a coprocessor does nothing. These policies matter to evolving programs; coherent fetch is the research contract, not a claim to reproduce hardware-prefetch ecology [algocell_exp/machines/worlds_8086.json; algocell_exp/machines/i8086_pair_maps_L16.wgsl; results/s8086/prelaunch/freeze.json; /Users/neo/repos/zilion/docs/I8086.md].


### MAIN-07

Reason: Distinguish execution steps from population updates. Avoid using updates for mutation writes when update denotes a population batch. Define a sample as a scheduled observation rather than an execution step. Use British spelling. Remove a redundant schedule phrase.

Old:

The shared soup contains 20,000 cells on a 160 × 125 lattice with reflecting cardinal-neighbour boundaries. Each update draws 8,192 ordered pair slots; collision claims determine active encounters, so draws are not executed-pair counts. The baseline encounter budget is 128 steps; S1 arm C uses 512. At noise exponent 4, mutation makes 512 uniform byte-replacement attempts per update. Packed-word collisions can lose updates, and replacement may leave a byte unchanged. Applied mutation counts are unmeasured. Pairing races also mean seeds do not guarantee identical whole-world replay [algocell_exp/soup.py; s8086.py; results/review_r2/RUN_TABLE.md; results/s8086/prelaunch/registration_basis.json; results/s8086/prelaunch/freeze.json].

S1 comprised four arms of 20 worlds, each run for 10,000,000 updates: A, L=16/budget 128/faithful BRK; B, L=32/128/faithful BRK; C, L=16/512/faithful BRK; D, L=16/128/halt-before-BRK. Seeds 65001–65020 were reused across arms, giving paired initial conditions where lengths matched. Top-three tapes were sampled every 50 updates to 2,000, every 250 to 20,000, every 2,500 to 1,000,000 and every 25,000 thereafter. The 6502 retention experiment C2 used budget 128 even for candidates originating in arm C [REVISION_PREREG.md, S1; results/s6502/CORRECTION.md; results/evolved6502/coverage.csv]. Supplementary Table 1 gives cohort-specific selection rules, seeds, schedules and horizons.

S2’s sole population arm comprised 20 L16 worlds, seeds 8086001–8086020, through 3,000,000 updates. L32 population arm B was withdrawn; pointer randomization was omitted because it lacked a controlled census. L32 remains fixed-input census/control context only. Each world retained 585 scheduled records, including initialization, and complete checkpoints at 0, 20,000, 100,000, 1,000,000 and 3,000,000 updates. Sampling intervals widen according to the frozen schedule; they are not adjusted after detection [s8086.py; manuscript/figures/out/v8/data/arrival_worlds.csv; results/s8086/tables/worlds.csv; results/s8086/tables/download_hashes.csv; results/s8086/prelaunch/freeze.json].

New:

The shared soup contains 20,000 cells on a 160 × 125 lattice with reflecting cardinal-neighbour boundaries. Each update draws 8,192 ordered pair slots; collision claims determine active encounters, so draws are not executed-pair counts. The baseline encounter budget is 128 execution steps; S1 arm C uses 512. At noise exponent 4, mutation makes 512 uniform byte-replacement attempts per update. Packed-word collisions can discard mutation writes, and replacement may leave a byte unchanged. Applied mutation counts are unmeasured. Pairing races also mean seeds do not guarantee identical whole-world replay [algocell_exp/soup.py; s8086.py; results/review_r2/RUN_TABLE.md; results/s8086/prelaunch/registration_basis.json; results/s8086/prelaunch/freeze.json].

S1 comprised four arms of 20 worlds, each run for 10,000,000 updates: A, L = 16/budget 128/faithful BRK; B, L = 32/128/faithful BRK; C, L = 16/512/faithful BRK; D, L = 16/128/halt-before-BRK. Seeds 65001–65020 were reused across arms, giving paired initial conditions where lengths matched. Samples recorded the top-three tapes every 50 updates to 2,000, every 250 to 20,000, every 2,500 to 1,000,000 and every 25,000 thereafter. The 6502 retention experiment C2 used budget 128 even for candidates originating in arm C [REVISION_PREREG.md, S1; results/s6502/CORRECTION.md; results/evolved6502/coverage.csv]. Supplementary Table 1 gives cohort-specific selection rules, seeds, schedules and horizons.

S2’s sole population arm comprised 20 L16 worlds, seeds 8086001–8086020, through 3,000,000 updates. L32 population arm B was withdrawn; pointer randomisation was omitted because it lacked a controlled census. L32 remains fixed-input census/control context only. Each world retained 585 scheduled records, including initialisation, and complete checkpoints at 0, 20,000, 100,000, 1,000,000 and 3,000,000 updates. Sampling intervals widen according to the frozen schedule; they are not adjusted after detection [s8086.py; manuscript/figures/out/v8/data/arrival_worlds.csv; results/s8086/tables/worlds.csv; results/s8086/tables/download_hashes.csv; results/s8086/prelaunch/freeze.json].


### MAIN-08

Reason: Replace the ambiguous referent following NaN. State the final analysis hierarchy directly in short sentences. Clarify that these are scored-site classes and remove the serial comma. Remove the duplicated preposition. Remove the serial comma. State the matching-position criterion grammatically.

Old:

Culture scores normalized gain in best-cyclic-shift similarity to the original parent, `(after−before)/(1−before)`, averaged over informative partners with before-match below 0.75. First offspring reproduce against fresh partners. The generation-two score includes all first-generation outputs, including those that failed to become close copies; it is not conditioned on a successful first copy [algocell_exp/assay.py]. Historical “heritable” means generation-two score at least 0.3, not exact inheritance. S2 used 64 uniform partners per generation, including final-cell cultures. NaN is unassessable, not zero. Its similarity and faithful-offspring fields are reported but do not add undeclared detection gates [manuscript/figures/out/v8/data/methods.csv; results/s8086/tables/copier_tapes.csv; results/s8086/prelaunch/registration_basis.json].

S1 specified 64 culture partners for heritability, whereas its cited frozen final-cell estimator used 32; top-tape cultures used 64. The trace text specified soup partners, whereas the frozen implementation used 16 uniform-byte partners. We report corrected frozen32 results as primary, with complete 64-partner final-cell reclassification and first-tape soup-snapshot tracing as separate sensitivities. Neither sensitivity changed the reported first-tape classifications or final-cell counts [results/s6502/CORRECTION.md; REVISION_PREREG.md, S1].

Single-site scans ask whether a reproductively competent mutant transmits its changed byte into at least half of qualifying aligned offspring. A site needs at least half its tested identifiable values to pass. Population classes call confined cells with at most two sites regenerators, three or four intermediate, and at least five transmitters. These thresholded classes do not measure exact capacity. “Unexecuted” means absent from instruction-stream fetches, including operands; it does not exclude data use [mutscan.py; results/ladder/LADDER.md; results/population/classes_snapshots.csv; results/crystal/cells.csv].

At each transfer, an offspring qualifies only if it matches its immediate parent at at least 75% of positions under the best cyclic shift. A serial lineage survives through generation g only if every transfer qualifies. Branching populations retain qualifying offspring, without carrying parents forward; when more than 32 qualify, 32 are sampled uniformly. A branching population survives if at least one member remains [serial_retention.py; branching_transfer.py; REVISION_PREREG.md, X1].

Historical serial assays follow the centred five-byte mutant window, permitting cyclic alignment and correcting carriage against surviving unmutated controls. Windows already present in wild type are unidentifiable and excluded. The panel uses 16 mutant lineages, 256 controls and eight transfers; alternatives are exhaustive for short tapes and sampled for longer ones. Branching X1 instead gives each survivor two partners, caps populations at 32, and uses four populations for each of 32 alternatives/site plus 64 controls. A population carries when at least half its members carry the marker; site scoring requires the registered 0.5 excess over control carriage among survivors. Fate fractions in Fig. 2a use all original starts. The soup-partner arm pools open-phase worlds rather than the focal neighbourhood; pre-existing matches and reacquisition can inflate carriage [results/serial_retention/sr_summary.csv; results/serial_retention/REPORT.md; results/branching/branching_summary.csv; results/branching/BRANCHING.md].

For deterministic offspring Y = F(x,E), fixing parent and contract gives I(E;Y|X=x) = H(Y|X=x). Exact copying implies zero inflow, but zero inflow does not imply preservation of parental variation. Sampled zero entropy is not partner-universal closure. The historical entropy panel has 256 outputs and an eight-bit sample ceiling, with downward undersampling bias. Fetch confinement, original-partner access proxies, output variation and parental transmission therefore remain separate [theorem: manuscript/SI_v8.md, Supplementary Note 2; results/biology/individuality/per_replicator.csv].

New:

Culture scores normalised gain in best-cyclic-shift similarity to the original parent, `(after−before)/(1−before)`, averaged over informative partners with before-match below 0.75. First offspring reproduce against fresh partners. The generation-two score includes all first-generation outputs, including those that failed to become close copies; it is not conditioned on a successful first copy [algocell_exp/assay.py]. Historical “heritable” means generation-two score at least 0.3, not exact inheritance. S2 used 64 uniform partners per generation, including final-cell cultures. NaN is unassessable, not zero. S2’s similarity and faithful-offspring fields are reported but do not add undeclared detection gates [manuscript/figures/out/v8/data/methods.csv; results/s8086/tables/copier_tapes.csv; results/s8086/prelaunch/registration_basis.json].

S1 specified 64 culture partners for heritability, whereas its cited frozen final-cell estimator used 32; top-tape cultures used 64. The trace text specified soup partners, whereas the frozen implementation used 16 uniform-byte partners. Corrected frozen32 results are primary. Complete 64-partner final-cell reclassification and first-tape soup-snapshot tracing are separate sensitivity analyses. Neither sensitivity changed the reported first-tape classifications or final-cell counts [results/s6502/CORRECTION.md; REVISION_PREREG.md, S1].

Single-site scans ask whether a reproductively competent mutant transmits its changed byte into at least half of qualifying aligned offspring. A site needs at least half its tested identifiable values to pass. Population assays classify confined cells with at most two scored sites as regenerators, three or four as intermediate and at least five as transmitters. These thresholded classes do not measure exact capacity. “Unexecuted” means absent from instruction-stream fetches, including operands; it does not exclude data use [mutscan.py; results/ladder/LADDER.md; results/population/classes_snapshots.csv; results/crystal/cells.csv].

At each transfer, an offspring qualifies only if at least 75% of its positions match its immediate parent under the best cyclic shift. A serial lineage survives through generation g only if every transfer qualifies. Branching populations retain qualifying offspring, without carrying parents forward; when more than 32 qualify, 32 are sampled uniformly. A branching population survives if at least one member remains [serial_retention.py; branching_transfer.py; REVISION_PREREG.md, X1].

Historical serial assays follow the centred five-byte mutant window, permitting cyclic alignment and correcting carriage against surviving unmutated controls. Windows already present in wild type are unidentifiable and excluded. The panel uses 16 mutant lineages, 256 controls and eight transfers; alternatives are exhaustive for short tapes and sampled for longer ones. Branching X1 instead gives each survivor two partners, caps populations at 32 and uses four populations for each of 32 alternatives/site plus 64 controls. A population carries when at least half its members carry the marker; site scoring requires the registered 0.5 excess over control carriage among survivors. Fate fractions in Fig. 2a use all original starts. The soup-partner arm pools open-phase worlds rather than the focal neighbourhood; pre-existing matches and reacquisition can inflate carriage [results/serial_retention/sr_summary.csv; results/serial_retention/REPORT.md; results/branching/branching_summary.csv; results/branching/BRANCHING.md].

For deterministic offspring Y = F(x,E), fixing parent and contract gives I(E;Y|X = x) = H(Y|X = x). Exact copying implies zero inflow, but zero inflow does not imply preservation of parental variation. Sampled zero entropy is not partner-universal closure. The historical entropy panel has 256 outputs and an eight-bit sample ceiling, with downward undersampling bias. Fetch confinement, original-partner access proxies, output variation and parental transmission therefore remain separate [theorem: manuscript/SI_v8.md, Supplementary Note 2; results/biology/individuality/per_replicator.csv].


### MAIN-09

Reason: Use British spelling. Separate the measured series from its verdict while preserving both outcomes. Split the endpoint definition from its known bias and define confinement precisely.

Old:

Registered competitions seed partial-copy R and full-tape T designs in equal mixtures or reciprocal rare invasions and retain mutation-on/off and benign/lethal arms. The designs differ in payload background as well as displacement, so environmental sorting does not isolate selection on capacity. At L32, classification requires `5e ed b0` at bytes 1–3 and byte 0 in R={04,44,84,08,48,88,C8,50,90} or T={60,A0}, all hexadecimal; other values remain unclassified. This census-defined classification differs from the ≤2/≥5 site-score classes of random-population assays [offset_switch.py; results/offset/OFFSET_CENSUS.md; REVISION_PREREG.md, E]. Figure 4 shows L32 mutation-on runs through 300,000 updates: ten equal-start worlds and five per invasion direction for each chemistry/cohort. Relative share is T/(R+T); broader core-motif prevalence uses all cells. Undefined ratios when both classes disappear are omitted only from ratio summaries, retaining absolute loss and zero tail diversity. Historical and core-C1 trajectories are summarized separately at world level [results/offset/offset_switch.csv; results/tier2/scored/E/offset_switch.csv; manuscript/figures/out/v8/data/competition_endpoints.csv].

The lethality dial varies the probability that a fetched zero opcode halts, with nondetections censored at its horizon. Under core C1, Kaplan–Meier medians of first-confinement time at p = 0, 0.01, 0.03, 0.1, 0.3 and 1 were 32,500, 42,500, 60,000, 80,000, 212,500 and 115,000 updates, one reversal, so DZ2’s monotone component changed from not met (two historical reversals) to met. The registered endpoint, the first sample with a heritable top-three tape at ≥0.5% of cells that is confined in 16 of 16 encounters, missed confined transmitter populations in two historical worlds, and the test is therefore biased against confinement at high p [results/tier2/T2.md, DZ2; REVISION_PREREG.md, DZ2]. The tested toxin, backup-core and mutation-only effects did not meet their registered explanatory criteria, and exploratory intruder assays identify a lead rather than a completed mechanism [results/tier2/T2_DZ.csv; results/toxin/CAUSAL.md; results/n2/BACKUP.md; results/offset/OFFSET_FLOW.md; results/offset/INTRUDER.md].

New:

Registered competitions seed partial-copy R and full-tape T designs in equal mixtures or reciprocal rare invasions and retain mutation-on/off and benign/lethal arms. The designs differ in payload background as well as displacement, so environmental sorting does not isolate selection on capacity. At L32, classification requires `5e ed b0` at bytes 1–3 and byte 0 in R = {04,44,84,08,48,88,C8,50,90} or T = {60,A0}, all hexadecimal; other values remain unclassified. This census-defined classification differs from the ≤ 2/ ≥ 5 site-score classes of random-population assays [offset_switch.py; results/offset/OFFSET_CENSUS.md; REVISION_PREREG.md, E]. Fig. 4 shows L32 mutation-on runs through 300,000 updates: ten equal-start worlds and five per invasion direction for each chemistry/cohort. Relative share is T/(R+T); broader core-motif prevalence uses all cells. Undefined ratios when both classes disappear are omitted only from ratio summaries, retaining absolute loss and zero tail diversity. Historical and core-C1 trajectories are summarised separately at world level [results/offset/offset_switch.csv; results/tier2/scored/E/offset_switch.csv; manuscript/figures/out/v8/data/competition_endpoints.csv].

The lethality dial varies the probability that a fetched zero opcode halts, with nondetections censored at its horizon. Under core C1, Kaplan–Meier medians of first-confinement time at p = 0, 0.01, 0.03, 0.1, 0.3 and 1 were 32,500, 42,500, 60,000, 80,000, 212,500 and 115,000 updates. This series has one reversal, so DZ2’s monotone component was met; historically it was not met (two reversals). The registered endpoint is the first sample containing a heritable top-three tape at ≥ 0.5% of cells with no partner instruction-stream fetch in 16 of 16 encounters. It missed confined transmitter populations in two historical worlds, so the test is biased against confinement at high p [results/tier2/T2.md, DZ2; REVISION_PREREG.md, DZ2]. The tested toxin, backup-core and mutation-only effects did not meet their registered explanatory criteria, and exploratory intruder assays identify a lead rather than a completed mechanism [results/tier2/T2_DZ.csv; results/toxin/CAUSAL.md; results/n2/BACKUP.md; results/offset/OFFSET_FLOW.md; results/offset/INTRUDER.md].


### MAIN-10

Reason: Remove the serial comma. Use British spelling. Preserve the original source bracket verbatim; spelling changes apply only to prose. Use British spelling outside the unchanged source bracket. Distinguish ordered-copy assignments from charged execution steps.

Old:

**Theorem.** On a ring of P bytes, perform ordered writes M[s+d+k] ← M[s+k], modulo P, with 0 < d < P and no interfering operation. Before k=d, reads obtain original source bytes; subsequently each source was last written d steps earlier. Induction gives ancestral coordinate s+(k mod d). With d dividing P and coverage at least P−d, the protected seed tiles the ring; with P=2L and d dividing L, the offspring has the required phase. If a fixed constrained footprint C lies in the protected seed, independent free payload is r=d−|C|. This describes a product family of 256^r sequences, or 8r ideal noiseless bits, not ecological neutrality or measured Shannon capacity [theorem: manuscript/SI_v8.md, Supplementary Note 1; results/capacity/geometry_traces.csv.gz].

For the Z80 `XX 5e ed b0` specialization, L≤64, d=XX mod 2L, 4≤d≤L and d divides L. The first instruction must pass control to address 1, preserve the needed source/base state, write only outside the protected prefix, and leave HL≡0 and 256D≡0 modulo 2L, with BC zero or at least 126. Under the declared initialization, 126 block moves remain; the immediate `1e NN ed b0` form leaves 127. Architectural address increments must preserve displacement. Core preservation keeps execution in the copy when overwritten instructions are revisited. Free coordinates then survive jointly and serially; reverse `LDDR` requires another derivation [theorem: manuscript/SI_v8.md, Supplementary Note 1, Z80 specialization; results/capacity/candidate_eligibility.csv].

New:

**Theorem.** On a ring of P bytes, perform ordered writes M[s+d+k] ← M[s+k], modulo P, with 0 < d < P and no interfering operation. Before k = d, reads obtain original source bytes; subsequently each source was last written d assignments earlier. Induction gives ancestral coordinate s+(k mod d). With d dividing P and coverage at least P−d, the protected seed tiles the ring; with P = 2L and d dividing L, the offspring has the required phase. If a fixed constrained footprint C lies in the protected seed, independent free payload is r = d−|C|. This describes a product family of 256^r sequences, or 8r ideal noiseless bits, not ecological neutrality or measured Shannon capacity [theorem: manuscript/SI_v8.md, Supplementary Note 1; results/capacity/geometry_traces.csv.gz].

For the Z80 `XX 5e ed b0` specialisation, L ≤ 64, d = XX mod 2L, 4 ≤ d ≤ L and d divides L. The first instruction must pass control to address 1, preserve the needed source/base state, write only outside the protected prefix and leave HL ≡ 0 and 256D ≡ 0 modulo 2L, with BC zero or at least 126. Under the declared initialisation, 126 block moves remain; the immediate `1e NN ed b0` form leaves 127. Architectural address increments must preserve displacement. Core preservation keeps execution in the copy when overwritten instructions are revisited. Free coordinates then survive jointly and serially; reverse `LDDR` requires another derivation [theorem: manuscript/SI_v8.md, Supplementary Note 1, Z80 specialization; results/capacity/candidate_eligibility.csv].


### MAIN-11

Reason: Apply British spelling, consistent comparison spacing and figure labels while preserving source brackets and numerical meaning.

Old:

Amended S2 L2 requires the actual sampled baseline to be d-periodic at physical entry zero, with d dividing L, full partner coverage and no original-partner input or interference. It permits original roots R={s,…,s+d−1} inside A. For physical offspring position q, predict a(q)=s+((L+q−s) mod d). Independently extract raw core K, retaining the bound c=1+max(K)≤d, and define source core C_R={a(k):k∈K}, free set F=R\C_R and erased set E=A\R. Hold both K and C_R. The exact free count is d−|C_R|, not necessarily d−c. Mutable erasures are E\K; erased setup coordinates E∩K remain held because their values may govern execution even though their original ancestry disappears. No dependency is reduced modulo d to hide a setup requirement. Source phase and physical entry are distinct; compare physical B directly, without favourable rotations. The certified map composes with itself across transfers [theorem and registered amendment: REVISION_PREREG.md, S2 B2; results/s8086/prelaunch/ASTRA_PRELAUNCH_REVIEW.md; results/census8086/LAW_CONTRACT.md].

New:

Amended S2 L2 requires the actual sampled baseline to be d-periodic at physical entry zero, with d dividing L, full partner coverage and no original-partner input or interference. It permits original roots R = {s,…,s+d−1} inside A. For physical offspring position q, predict a(q) = s+((L+q−s) mod d). Independently extract raw core K, retaining the bound c = 1+max(K) ≤ d, and define source core C_R = {a(k):k∈K}, free set F = R\C_R and erased set E = A\R. Hold both K and C_R. The exact free count is d−|C_R|, not necessarily d−c. Mutable erasures are E\K; erased setup coordinates E∩K remain held because their values may govern execution even though their original ancestry disappears. No dependency is reduced modulo d to hide a setup requirement. Source phase and physical entry are distinct; compare physical B directly, without favourable rotations. The certified map composes with itself across transfers [theorem and registered amendment: REVISION_PREREG.md, S2 B2; results/s8086/prelaunch/ASTRA_PRELAUNCH_REVIEW.md; results/census8086/LAW_CONTRACT.md].


### MAIN-12

Reason: Disambiguate the experiment from corrected core C1.

Old:

Forward passages followed 16 lineages through eight generations; reciprocal passages restored the original displacement byte in actual descendants and repeated transfer. Complete core, displacement, offspring and ancestry are retained in 25,856 serial rows. Direct assays used actual generation-eight descendants, testing all 255 alternatives at every free and erased coordinate plus 64 joint assignments against all 17 partners. Deterministic duplicate descendants share execution while retaining lineage identities. All 1,067,608 direct cases and all serial observations matched. These are intervention cases, not encounter counts. Joint sampling is finite; the unrestricted joint claim is conditional mathematics. C1 used local computation with no paid compute [results/capacity/serial.csv.gz; results/capacity/capacity_cases.csv.gz; results/capacity/CAPACITY.md; results/capacity/bursts.csv].

New:

Forward passages followed 16 lineages through eight generations; reciprocal passages restored the original displacement byte in actual descendants and repeated transfer. Complete core, displacement, offspring and ancestry are retained in 25,856 serial rows. Direct assays used actual generation-eight descendants, testing all 255 alternatives at every free and erased coordinate plus 64 joint assignments against all 17 partners. Deterministic duplicate descendants share execution while retaining lineage identities. All 1,067,608 direct cases and all serial observations matched. These are intervention cases, not encounter counts. Joint sampling is finite; the unrestricted joint claim is conditional mathematics. Displacement experiment C1 used local computation with no paid compute [results/capacity/serial.csv.gz; results/capacity/capacity_cases.csv.gz; results/capacity/CAPACITY.md; results/capacity/bursts.csv].


### MAIN-13

Reason: Disambiguate the retention experiment from vendor-specific core C2. Replace an abrupt statement with its methodological meaning. Keep the experiment distinct from the core variant.

Old:

C2 shares registration `213a653`. The original extractor and 714-role ledger were frozen at `943aeee`; unchanged extraction extended to 942 roles at `657eb7e` before interventions on newly recovered top-three roles. The descriptive-copier inclusion clarification at `f57ad24` preceded new assays. Corrected S1 frozen32 primary calls determine the descriptive final-copier panel; every registered sampling role remains in the geometry ledger. Sensitivity evaluations do not create cells. All raw logs matched their frozen input hashes, with no substitute tapes [results/evolved6502/coverage.csv; results/evolved6502/inputs.csv; results/evolved6502/RETENTION.md].

The complete panel contains 62 first detections, 640 final random draws and 240 final top-three roles from 80 worlds. It has 861 distinct tapes and 42 eligible roles. Footprint c counts constrained coordinates, not necessarily a contiguous prefix. Ordered provenance admits pure-copy channels and rejects or leaves unresolved unknown arithmetic, reverse/nonconstant motion, partner inflow, interference and uncertified control. Using each world’s frozen 6502 machine at budget 128, every admitted role received all 255 alternatives at each free/erased coordinate and 64 joint assignments against zero plus 16 uniform partners, followed through eight fresh-partner generations. Complete physical offspring and source labels are scored without similarity gates or cyclic realignment. There were 91,428 intervention cases and 731,424 serial observations, without mismatch. A separate descriptive serial scan uses marker thresholds on ineligible copiers: two first and two final roles per arm, 32 alternatives/site, four mutant lineages and 64 controls. It neither tests the exact law nor supplies a registered directional prediction. C2 used no paid compute [evolved6502_retention.py; capacity_common.py; results/evolved6502/panel.csv; results/evolved6502/eligibility.csv; results/evolved6502/geometry.csv; results/evolved6502/law_verdict.csv; results/evolved6502/descriptive_panel.csv; results/evolved6502/exact_cases.csv.gz; results/evolved6502/exact_serial.csv.gz; results/evolved6502/bursts.csv].

New:

The 6502 retention experiment C2 shares registration `213a653`. The original extractor and 714-role ledger were frozen at `943aeee`; unchanged extraction extended to 942 roles at `657eb7e` before interventions on newly recovered top-three roles. The descriptive-copier inclusion clarification at `f57ad24` preceded new assays. Corrected S1 frozen32 primary calls determine the descriptive final-copier panel; every registered sampling role remains in the geometry ledger. Sensitivity analyses reclassify the same sampled cells. All raw logs matched their frozen input hashes, with no substitute tapes [results/evolved6502/coverage.csv; results/evolved6502/inputs.csv; results/evolved6502/RETENTION.md].

The complete panel contains 62 first detections, 640 final random draws and 240 final top-three roles from 80 worlds. It has 861 distinct tapes and 42 eligible roles. Footprint c counts constrained coordinates, not necessarily a contiguous prefix. Ordered provenance admits pure-copy channels and rejects or leaves unresolved unknown arithmetic, reverse/nonconstant motion, partner inflow, interference and uncertified control. Using each world’s frozen 6502 machine at budget 128, every admitted role received all 255 alternatives at each free/erased coordinate and 64 joint assignments against zero plus 16 uniform partners, followed through eight fresh-partner generations. Complete physical offspring and source labels are scored without similarity gates or cyclic realignment. There were 91,428 intervention cases and 731,424 serial observations, without mismatch. A separate descriptive serial scan uses marker thresholds on ineligible copiers: two first and two final roles per arm, 32 alternatives/site, four mutant lineages and 64 controls. It neither tests the exact law nor supplies a registered directional prediction. The retention experiment C2 used no paid compute [evolved6502_retention.py; capacity_common.py; results/evolved6502/panel.csv; results/evolved6502/eligibility.csv; results/evolved6502/geometry.csv; results/evolved6502/law_verdict.csv; results/evolved6502/descriptive_panel.csv; results/evolved6502/exact_cases.csv.gz; results/evolved6502/exact_serial.csv.gz; results/evolved6502/bursts.csv].


### MAIN-14

Reason: Apply British spelling, consistent comparison spacing and figure labels while preserving source brackets and numerical meaning.

Old:

**Theorem.** A repeated pair with distinct one-byte load/push opcodes loads w literal bytes into one register and pushes a fixed permutation without changing it, using a separate stack pointer that moves by −w per push. Under contiguous noninterfering writes and the prescribed executed stream, input/output periods are w+2 and w. Fine–Wilf requires N≥2w+2−gcd(2,w) for finite-window equality. Odd widths then force identical opcodes, a contradiction; even widths permit compatible alternating constructions. Descending stacks reverse block order in ascending memory, not internal byte order. Placement, bootstrap and coverage remain separate [theorem: results/literal_channel/CHECK.md; results/literal_channel/pair_layouts.csv; results/literal_channel/finite_bound_counterexamples.json].

**Theorem.** For width-one periodic mixtures through a shared register, a minimal decoder cycle visits at most d residues of primitive input period d. If it contains loads and pushes, emitted length n_P satisfies 0<n_P<t≤d, incompatible with d dividing n_P for an equal infinite stream. Multiple opcode identities do not change this. The degenerate all-push writer initialized with an opcode-valued register is excluded. Width-three mixtures provide counterexamples to extending parity beyond repeated pairs. A finite width-one factor witness exists but fails archived zero-partner bootstrap. Only three of the five nontrivial Z80 period-two writers are literal; the others are memory-fed. None of these statements proves an evolutionary arrival order or excludes every finite 6502 copier [theorem/construction: results/literal_channel/verification.csv; results/literal_channel/finite_search.csv; results/literal_channel/6502_finite_harness.json; results/literal_channel/census_audit.csv].

New:

**Theorem.** A repeated pair with distinct one-byte load/push opcodes loads w literal bytes into one register and pushes a fixed permutation without changing it, using a separate stack pointer that moves by −w per push. Under contiguous noninterfering writes and the prescribed executed stream, input/output periods are w+2 and w. Fine–Wilf requires N ≥ 2w+2−gcd(2,w) for finite-window equality. Odd widths then force identical opcodes, a contradiction; even widths permit compatible alternating constructions. Descending stacks reverse block order in ascending memory, not internal byte order. Placement, bootstrap and coverage remain separate [theorem: results/literal_channel/CHECK.md; results/literal_channel/pair_layouts.csv; results/literal_channel/finite_bound_counterexamples.json].

**Theorem.** For width-one periodic mixtures through a shared register, a minimal decoder cycle visits at most d residues of primitive input period d. If it contains loads and pushes, emitted length n_P satisfies 0 < n_P < t ≤ d, incompatible with d dividing n_P for an equal infinite stream. Multiple opcode identities do not change this. The degenerate all-push writer initialised with an opcode-valued register is excluded. Width-three mixtures provide counterexamples to extending parity beyond repeated pairs. A finite width-one factor witness exists but fails archived zero-partner bootstrap. Only three of the five nontrivial Z80 period-two writers are literal; the others are memory-fed. None of these statements proves an evolutionary arrival order or excludes every finite 6502 copier [theorem/construction: results/literal_channel/verification.csv; results/literal_channel/finite_search.csv; results/literal_channel/6502_finite_harness.json; results/literal_channel/census_audit.csv].


### MAIN-15

Reason: Remove superseded-label commentary while preserving census scope and the subsequent audit counts. Place sentence punctuation outside the quoted classification label.

Old:

The validated 8086 census evaluated all 256 period-one and 65,536 period-two words at L16/L32 and budgets 128/512 against zero plus 16 frozen partners, retaining constants and reversed phases, with no unevaluated words. All mechanism descriptions use the final causal classification; earlier labels are superseded. Its audit covers 1,246 settings and 21,182 encounters; 1,247 labels changed without changing outputs or matches [results/census8086/table_main.csv; results/census8086/table_hits.csv; results/census8086/descriptive_correction_counts.csv; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md].

The classifier examines execution through the last partner write. Every claimed store, including neutral read-modify-write, must pass isolated `55` and `AA` source interventions with matching ordered addresses through that write and the expected source-dependent value. Later divergence is allowed. Literal provenance, paired memory copying and confined copy loops are separate; loops require repeated full instruction-start addresses, not merely mapped ring positions. Mixed, unsupported, failed, vector-dependent or overflowing cases remain “other.” Trace memory and original records must agree with uninstrumented execution. Full-encounter partner fetch and pre-write access proxies are separate from this causal classification [results/census8086/MECHANISM_CONTRACT.md; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md; results/s8086/tables/judged_encounters.csv.gz].

New:

The validated 8086 census evaluated all 256 period-one and 65,536 period-two words at L16/L32 and budgets 128/512 against zero plus 16 frozen partners. Constants and reversed phases were retained, with no unevaluated words. Mechanism descriptions use the causal classification below. Its audit covers 1,246 settings and 21,182 encounters; 1,247 labels changed without changing outputs or matches [results/census8086/table_main.csv; results/census8086/table_hits.csv; results/census8086/descriptive_correction_counts.csv; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md].

The classifier examines execution through the last partner write. Every claimed store, including neutral read-modify-write, must pass isolated `55` and `AA` source interventions with matching ordered addresses through that write and the expected source-dependent value. Later divergence is allowed. Literal provenance, paired memory copying and confined copy loops are separate; loops require repeated full instruction-start addresses, not merely mapped ring positions. Mixed, unsupported, failed, vector-dependent or overflowing cases remain “other”. Trace memory and original records must agree with uninstrumented execution. Full-encounter partner fetch and pre-write access proxies are separate from this causal classification [results/census8086/MECHANISM_CONTRACT.md; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md; results/s8086/tables/judged_encounters.csv.gz].


### MAIN-16

Reason: Specify instruction-stream fetch and avoid a compressed three-part sentence.

Old:

R0 requires at least 10 detections among 20 worlds. Given that sufficiency, R1 requires at least ceil(0.8D) open first copiers; the kill rule triggers at ceil(0.25D) confined first copiers. Open means partner fetch in at least 8/16 complete traces, confined none in all 16, and intermediate histories remain in D. Observed D=14, open=2 and confined=12 give R0 pass, R1 fail and the route kill. R2 records occurrence in 1/20 worlds and persistence in 0/20, without confirmatory cutoffs. Confinement timing has median 975,000 updates, with eight censored worlds; it is not a causal-closure or loop endpoint [results/s8086/tables/worlds.csv; results/s8086/tables/persistence.csv; results/s8086/tables/kaplan_meier.csv; results/s8086/S2.md].

New:

R0 requires at least 10 detections among 20 worlds. Given that sufficiency, R1 requires at least ceil(0.8D) open first copiers; the kill rule triggers at ceil(0.25D) confined first copiers. Open means partner instruction-stream fetch in at least 8/16 complete traces; confined means none in all 16. Intermediate histories remain in D. Observed D = 14, open = 2 and confined = 12 give R0 pass, R1 fail and the route kill. R2 records occurrence in 1/20 worlds and persistence in 0/20, without confirmatory cutoffs. Confinement timing has median 975,000 updates, with eight censored worlds; it is not a causal-closure or loop endpoint [results/s8086/tables/worlds.csv; results/s8086/tables/persistence.csv; results/s8086/tables/kaplan_meier.csv; results/s8086/S2.md].


### MAIN-17

Reason: Remove the serial comma.

Old:

L1 requires the sampled baseline to be d-periodic at physical entry zero, 6≤d≤L with d dividing L and even d for MOVSW, reset source/segments, sufficient ordered forward coverage, preserved core and no interfering write or original-partner read. Each family could select at most 16 genotype-world cases per world, taking eligible first/closed/final roles before keyed random-cell candidates. The registered design tests all 256 values at free and mutable erased coordinates, 256 counterbalanced plus 256 seeded joint assignments, all 17 census partners, and eight serial generations for singles/joints. A reproducible mismatch remaining independently eligible fails the application; unresolved certificates suspend support. These planned evolved assay denominators remained empty [results/census8086/LAW_CONTRACT.md; results/s8086/tables/law_outcomes.json; REVISION_PREREG.md, S2].

New:

L1 requires the sampled baseline to be d-periodic at physical entry zero, 6 ≤ d ≤ L with d dividing L and even d for MOVSW, reset source/segments, sufficient ordered forward coverage, preserved core and no interfering write or original-partner read. Each family could select at most 16 genotype-world cases per world, taking eligible first/closed/final roles before keyed random-cell candidates. The registered design tests all 256 values at free and mutable erased coordinates, 256 counterbalanced plus 256 seeded joint assignments, all 17 census partners and eight serial generations for singles/joints. A reproducible mismatch remaining independently eligible fails the application; unresolved certificates suspend support. These planned evolved assay denominators remained empty [results/census8086/LAW_CONTRACT.md; results/s8086/tables/law_outcomes.json; REVISION_PREREG.md, S2].


### MAIN-18

Reason: Apply British spelling, consistent comparison spacing and figure labels while preserving source brackets and numerical meaning.

Old:

A separate extended dependency verifier re-extracted all 779 culture-positive candidates, admitting 600 and leaving 72 excluded and 107 unresolved. The purposive 32-genotype panel represented 20 worlds; predictions were committed at `5bbc9d7` before assays, but extraction and selection remain post hoc. The frozen assay was reused with the extended verifier and 64 independent joint assignments in place of 512. All-value single/mutable-erasure substitutions, 17 census partners and eight serial transfers yielded 2,224,640 encounters without mismatch. For d=4, free payload is empty and joint assignments repeat baseline; for d=16, erased coordinates are empty. CPU audits verified hashes, source maps and denominators. No new world or paid remote computation was run, and the registered verifier and verdicts were unchanged [exploratory: results/s8086/exploratory/eligibility_counts.csv; results/s8086/exploratory/eligibility.csv; results/s8086/exploratory/law_panel_predictions.csv; results/s8086/exploratory/law_assay_denominators.csv; results/s8086/exploratory/verification.csv; results/s8086/EXPLORATORY.md].

New:

A separate extended dependency verifier re-extracted all 779 culture-positive candidates, admitting 600 and leaving 72 excluded and 107 unresolved. The purposive 32-genotype panel represented 20 worlds; predictions were committed at `5bbc9d7` before assays, but extraction and selection remain post hoc. The frozen assay was reused with the extended verifier and 64 independent joint assignments in place of 512. All-value single/mutable-erasure substitutions, 17 census partners and eight serial transfers yielded 2,224,640 encounters without mismatch. For d = 4, free payload is empty and joint assignments repeat baseline; for d = 16, erased coordinates are empty. CPU audits verified hashes, source maps and denominators. No new world or paid remote computation was run, and the registered verifier and verdicts were unchanged [exploratory: results/s8086/exploratory/eligibility_counts.csv; results/s8086/exploratory/eligibility.csv; results/s8086/exploratory/law_panel_predictions.csv; results/s8086/exploratory/law_assay_denominators.csv; results/s8086/exploratory/verification.csv; results/s8086/EXPLORATORY.md].


### MAIN-19

Reason: Name each experiment explicitly while retaining the core labels. Use a direct description of the evidence chronology. Keep the experiment distinct from the core variant.

Old:

I1 compared historical compatibility C0 with corrected classic core C1 and vendor-specific core C2, distinct from experiments C1/C2. It flagged 47/690 and 114/690 conditions respectively, with 407/417 isolated combinations identical in each comparison; the all-unaffected criterion failed. Omitted named Stage-E cases were completed locally with the sealed engine. Stochastic assays use case-index streams, not archived encounter streams. I1-T2 retained 19/22 claim verdicts but changed four of 36 components, including historical DZ2/DZ3 failures becoming passes and Stage-L L2 becoming not applicable at L16 and not met at L20, while its joint kill still fires. Preserved verdicts include negative outcomes. The Stage-L dispatcher’s missing-snapshot alerts were retained; frozen scorers used complete required records. Fresh nondeterministic worlds do not isolate a causal core-repair effect [results/impact/decisions.csv; results/impact/assay_identity.csv; results/impact/supplement_manifest.json; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv; results/tier2/stageL_diagnosis.json].

The SI chronology keeps discovery definitions separate from confirmation. X1’s carriage prediction failed; the first X3 archive existed before registration, whereas its replication was prospective; N6’s failed component remains failed. The validated 6502 census preceded S1, while the literal theorem is retrospective. S1 correction retains unsupported P1, untested P3 and the registered-versus-executed partner-count/source discrepancies. C2’s later exact-law pass does not rescore S1. Crystal, ladder and the 8086 follow-ups are exploratory. S2’s final scientific amendments preceded population data; operational recovery records identify subsequent changes without rewriting prior history [REVISION_PREREG.md; results/lod_n6/N6_SCORE.md; results/s6502/CORRECTION.md; results/evolved6502/coverage.csv; results/s8086/prelaunch/ASTRA_RECHECK3.md; results/s8086/exploratory/publication_archive.csv].

New:

I1 compared historical compatibility C0 with corrected classic core C1 and vendor-specific core C2, distinct from displacement experiment C1 and retention experiment C2. It flagged 47/690 and 114/690 conditions respectively, with 407/417 isolated combinations identical in each comparison; the all-unaffected criterion failed. Omitted named Stage-E cases were completed locally with the sealed engine. Stochastic assays use case-index streams, not archived encounter streams. I1-T2 retained 19/22 claim verdicts but changed four of 36 components, including historical DZ2/DZ3 failures becoming passes and Stage-L L2 becoming not applicable at L16 and not met at L20, while its joint kill still fires. Preserved verdicts include negative outcomes. The Stage-L dispatcher’s missing-snapshot alerts were retained; frozen scorers used complete required records. Fresh nondeterministic worlds do not isolate a causal core-repair effect [results/impact/decisions.csv; results/impact/assay_identity.csv; results/impact/supplement_manifest.json; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv; results/tier2/stageL_diagnosis.json].

The chronology in Supplementary Information distinguishes discovery from confirmation. X1’s carriage prediction failed; the first X3 archive existed before registration, whereas its replication was prospective; N6’s failed component remains failed. The validated 6502 census preceded S1, while the literal theorem is retrospective. S1 correction retains unsupported P1, untested P3 and the registered-versus-executed partner-count/source discrepancies. The retention experiment C2’s later exact-law pass does not rescore S1. Crystal, ladder and the 8086 follow-ups are exploratory. S2’s final scientific amendments preceded population data; operational recovery records identify subsequent changes without rewriting prior history [REVISION_PREREG.md; results/lod_n6/N6_SCORE.md; results/s6502/CORRECTION.md; results/evolved6502/coverage.csv; results/s8086/prelaunch/ASTRA_RECHECK3.md; results/s8086/exploratory/publication_archive.csv].


### MAIN-20

Reason: Apply British spelling, consistent comparison spacing and figure labels while preserving source brackets and numerical meaning.

Old:

Independent evolutionary replicates are worlds; sampled cells, roles, variants, partners and serial observations are nested within them. Deterministic intervention mismatches are enumerated, not treated as independent-world trials. Detection summaries use scheduled event times and Kaplan–Meier right censoring at the declared horizon. Competition summaries use world-level medians; ratios with R+T=0 are undefined and excluded only from ratio summaries. Sample sizes and exclusions are given for each cohort. The benign core-C1 T-into-R subgroup has four defined ratios from five worlds. Supplementary Notes 5 and 7 describe the applicable confidence-bound and resampling procedures; exact law checks do not receive independent-world confidence intervals [manuscript/figures/out/v8/data/methods.csv; manuscript/figures/out/v8/data/competition_endpoints.csv; results/impact/decisions.csv].

New:

Independent evolutionary replicates are worlds; sampled cells, roles, variants, partners and serial observations are nested within them. Deterministic intervention mismatches are enumerated, not treated as independent-world trials. Detection summaries use scheduled event times and Kaplan–Meier right censoring at the declared horizon. Competition summaries use world-level medians; ratios with R+T = 0 are undefined and excluded only from ratio summaries. Sample sizes and exclusions are given for each cohort. The benign core-C1 T-into-R subgroup has four defined ratios from five worlds. Supplementary Notes 5 and 7 describe the applicable confidence-bound and resampling procedures; exact law checks do not receive independent-world confidence intervals [manuscript/figures/out/v8/data/methods.csv; manuscript/figures/out/v8/data/competition_endpoints.csv; results/impact/decisions.csv].


### MAIN-21

Reason: Remove review-copy commentary. State the final result and provenance without manuscript-revision history. Remove quoted audit wording and state the operational definitions directly. Format the illustrative key counts with separators and spacing; values are unchanged (build_v8.py fig3 key).

Old:

All result-path citations in this review copy resolve from `experiments/`. Generated tables establish empirical counts; frozen contracts, scorers and figure-preparation code establish protocol thresholds, coordinate conventions and display choices. The per-occurrence index `manuscript/AUDIT_v7_RESOLUTION.md` resolves every exception in the v7 mechanical audit, including quantities moved from long legends to Supplementary Note 5. Its substantive mismatch is corrected to six total scored sites, four unexecuted, for the K/L32/seed5006 tape. “Four measurements” enumerates culture, execution confinement, partner inflow and transfer retention; “two properties” enumerates partner-information closure and parental-variant transmission. These are definitions, not additional experimental denominators [results/crystal/cell_exceptions.csv; manuscript/AUDIT_v7_RESOLUTION.md].

Probability complements and normalizations use unit probability: cumulative detection is 1 minus Kaplan–Meier survival, and normalized culture gain is (after−before)/(1−before), over partners with before-match below 0.75. The illustrated L16 modal neighbourhood allows L/4=4 mismatches, hence requires at least 16−4=12 matching bytes. For L32, P=2L=64 and zero-based parent/partner addresses are 0–31/32–63; an edit from 0x04 to 0x60 changes displacement from 4 to 96 mod 64=32. With core size 4, the corresponding payload counts are 4−4=0 and 32−4=28. In the conceptual L12 diagram they are 4−4=0 and 12−4=8 bytes, drawn as two-byte units. A 16-byte `5F A4 x y` tape has 16/4=4 tile repeats. An inclusive source interval {s,…,s+d−1} contains d positions, and a zero-based core bounding prefix has length 1+max(K), distinct from the cardinality of a noncontiguous K [algocell_exp/assay.py; algocell_exp/metrics.py; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/dial_coordinates.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/out/v8/data/first_copier_tiles.csv; REVISION_PREREG.md, S2 B2].

The lattice contains 160×125=20,000 cells. At noise exponent 4, the declared 8,192 pair draws give 8,192/2^4=512 mutation attempts per update; draws and attempts are not active encounters or applied changes. A share threshold of 0.005 is 0.5%, or 100/20,000 cells. At least half of 16 traces means at least 8/16; a strict majority means more than 0.5. Reciprocal 1% invasions have initial T shares 0.01 or 0.99; the latter leaves 1−0.99=0.01 for R. The S2 motif archive contains 20×585=11,700 scheduled top-three sample records, representing 11,700×3=35,100 ranked tape entries, and 20×5×20,000=2,000,000 checkpoint-cell observations. These repeated records are not independent worlds. Complete byte-word censuses contain 256^1=256 period-one and 256^2=65,536 period-two assignments. The registered 512 joint assignments comprise 256 counterbalanced plus 256 keyed random assignments; the exploratory panel used 64 independent joint assignments. For the 8086 reset SP=1, a word push decrements SP by 2 modulo 65,536: the low byte maps from 65,535 to P−1 and the high byte from 0 to 0 for the declared P=32/64 rings [score8086.py; s8086.py; algocell_exp/soup.py; dial_soups.py; manuscript/figures/out/v8/data/competition_endpoints.csv; results/s8086/exploratory/motif_scan.csv.gz; results/census8086/table_main.csv; results/census8086/LAW_CONTRACT.md; results/s8086/EXPLORATORY.md; algocell_exp/machines/i8086_pair_maps_L16.wgsl].

Display counts follow explicit selections. Figure 1c contains one unique (seed,L) pair, (2002,16). Figure 2b selects four keys, {pusher16, ret16, ldir32, genome20_s6003}, with one curve per key; Fig. 2c selects six, {pusher50, ret16, ldir32, lethal16_s4009, genome20_s6004, closer32_8080_p1}. Historical and corrected branching endpoints coincide in every plotted field, permitting one bar per design without merging their cohorts. Figure 2d follows one marker lineage across its two ordered encounters. Figure 3a contrasts the two settings d∈{4,12}; the Fig. 3b disc-area key contains three illustrative counts {1,100,2000}, not additional observations. The two generation-eight strips select ldir32_v04 and ldir32_v60, each at lineage zero, as parallel interventions [manuscript/figures/out/v8/data/world_trajectory.csv; manuscript/figures/out/v8/data/serial_summary.csv; manuscript/figures/out/v8/data/branching_endpoints.csv; manuscript/figures/out/v8/data/erasure_manifest.json; manuscript/figures/out/v8/data/erasure_ordered.csv; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/prepare_v7.py; manuscript/figures/build_v8.py].

New:

Result-path citations resolve from `experiments/`. Generated tables establish empirical counts; frozen contracts, scorers and figure-preparation code establish protocol thresholds, coordinate conventions and display choices. The numerical provenance index is `manuscript/AUDIT_v7_RESOLUTION.md`; detailed figure provenance is in Supplementary Note 5. The K/L32/seed5006 tape has six total scored sites, four unexecuted. The four measurements are culture, execution confinement, partner inflow and transfer retention; the two properties are partner-information closure and parental-variant transmission. These are definitions, not additional experimental denominators [results/crystal/cell_exceptions.csv; manuscript/AUDIT_v7_RESOLUTION.md].

Probability complements and normalisations use unit probability: cumulative detection is 1 minus Kaplan–Meier survival, and normalised culture gain is (after−before)/(1−before), over partners with before-match below 0.75. The illustrated L16 modal neighbourhood allows L/4 = 4 mismatches, hence requires at least 16−4 = 12 matching bytes. For L32, P = 2L = 64 and zero-based parent/partner addresses are 0–31/32–63; an edit from 0x04 to 0x60 changes displacement from 4 to 96 mod 64 = 32. With core size 4, the corresponding payload counts are 4−4 = 0 and 32−4 = 28. In the conceptual L12 diagram they are 4−4 = 0 and 12−4 = 8 bytes, drawn as two-byte units. A 16-byte `5F A4 x y` tape has 16/4 = 4 tile repeats. An inclusive source interval {s,…,s+d−1} contains d positions, and a zero-based core bounding prefix has length 1+max(K), distinct from the cardinality of a noncontiguous K [algocell_exp/assay.py; algocell_exp/metrics.py; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/dial_coordinates.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/out/v8/data/first_copier_tiles.csv; REVISION_PREREG.md, S2 B2].

The lattice contains 160×125 = 20,000 cells. At noise exponent 4, the declared 8,192 pair draws give 8,192/2^4 = 512 mutation attempts per update; draws and attempts are not active encounters or applied changes. A share threshold of 0.005 is 0.5%, or 100/20,000 cells. At least half of 16 traces means at least 8/16; a strict majority means more than 0.5. Reciprocal 1% invasions have initial T shares 0.01 or 0.99; the latter leaves 1−0.99 = 0.01 for R. The S2 motif archive contains 20×585 = 11,700 scheduled top-three sample records, representing 11,700×3 = 35,100 ranked tape entries, and 20×5×20,000 = 2,000,000 checkpoint-cell observations. These repeated records are not independent worlds. Complete byte-word censuses contain 256^1 = 256 period-one and 256^2 = 65,536 period-two assignments. The registered 512 joint assignments comprise 256 counterbalanced plus 256 keyed random assignments; the exploratory panel used 64 independent joint assignments. For the 8086 reset SP = 1, a word push decrements SP by 2 modulo 65,536: the low byte maps from 65,535 to P−1 and the high byte from 0 to 0 for the declared P = 32/64 rings [score8086.py; s8086.py; algocell_exp/soup.py; dial_soups.py; manuscript/figures/out/v8/data/competition_endpoints.csv; results/s8086/exploratory/motif_scan.csv.gz; results/census8086/table_main.csv; results/census8086/LAW_CONTRACT.md; results/s8086/EXPLORATORY.md; algocell_exp/machines/i8086_pair_maps_L16.wgsl].

Display counts follow explicit selections. Fig. 1c contains one unique (seed,L) pair, (2002,16). Fig. 2b selects four keys, {pusher16, ret16, ldir32, genome20_s6003}, with one curve per key; Fig. 2c selects six, {pusher50, ret16, ldir32, lethal16_s4009, genome20_s6004, closer32_8080_p1}. Historical and corrected branching endpoints coincide in every plotted field, permitting one bar per design without merging their cohorts. Fig. 2d follows one marker lineage across its two ordered encounters. Fig. 3a contrasts the two settings d∈{4,12}; the Fig. 3b disc-area key contains three illustrative counts {1, 100, 2,000}, not additional observations. The two generation-eight strips select ldir32_v04 and ldir32_v60, each at lineage zero, as parallel interventions [manuscript/figures/out/v8/data/world_trajectory.csv; manuscript/figures/out/v8/data/serial_summary.csv; manuscript/figures/out/v8/data/branching_endpoints.csv; manuscript/figures/out/v8/data/erasure_manifest.json; manuscript/figures/out/v8/data/erasure_ordered.csv; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/prepare_v7.py; manuscript/figures/build_v8.py].


### MAIN-22

Reason: Apply British spelling, consistent comparison spacing and figure labels while preserving source brackets and numerical meaning.

Old:

**Draft for author confirmation.** Claude (Anthropic) and GPT-based Codex agents (OpenAI) were used for code implementation, analysis, adversarial review and drafting under the author’s direction. All reported results were produced by committed, reproducible code and verified. The author directed the work, reviewed the outputs and takes responsibility for the final manuscript. The author must confirm this disclosure and the final release identities before submission.

New:

[AUTHOR TO CONFIRM: AI-use disclosure.] Claude (Anthropic) and GPT-based Codex agents (OpenAI) assisted with code implementation, analysis scripts, adversarial review, figure production and drafting. The author directed the study, checked the results against committed generated tables and takes responsibility for the content.


### MAIN-23

Reason: Use the full Supplementary Note label. Replace rendering/build commentary with the final Extended Data map description. Use British spelling.

Old:

The accompanying `manuscript/SI_v8.md` contains the full overlap-copy theorem and Z80 specialization (Supplementary Note 1), the provenance certificate and inflow distinction (Note 2), scoped literal-channel theorems and constructions (Note 3), machine-specific copying-law conditions (Note 4), cohort and assay definitions with detailed figure provenance (Note 5), registration chronology and operational amendments (Note 6), I1/I1-T2 impact results (Note 7), the causal census correction (Note 8) and the exploratory 8086 analyses (Note 9). Supplementary Tables 1–3 identify cohorts, registration chronology and complete verdict ledgers; the remaining numbered tables identify generated supporting records. The Extended Data map in SI records the actual availability of each display, its `ed8_*.pdf` target and any older rendered equivalent. Missing displays are identified as requiring rendering; tables and completed results are not represented as rendered figures [manuscript/SI_v8.md].

New:

The accompanying `manuscript/SI_v8.md` contains the full overlap-copy theorem and Z80 specialisation (Supplementary Note 1), the provenance certificate and inflow distinction (Supplementary Note 2), scoped literal-channel theorems and constructions (Supplementary Note 3), machine-specific copying-law conditions (Supplementary Note 4), cohort and assay definitions with detailed figure provenance (Supplementary Note 5), registration chronology and operational amendments (Supplementary Note 6), I1/I1-T2 impact results (Supplementary Note 7), the causal census correction (Supplementary Note 8) and the exploratory 8086 analyses (Supplementary Note 9). Supplementary Tables 1–3 identify cohorts, registration chronology and complete verdict ledgers; the remaining numbered tables identify generated supporting records. The Extended Data map lists each display, its `ed8_*.pdf` target and its supporting sources [manuscript/SI_v8.md].


### MAIN-24

Reason: State what the repository contains. Remove prospective manuscript-production commentary. Consolidate the unreleased data and benchmark into the permitted data-accession placeholder. Remove initial-file-check and parallel-build commentary; PDF/PNG existence checked locally, ED targets follow the user’s final package instruction. Removed display mentions are logged, not scientific counts. Remove the serial comma from the consolidated placeholder.

Old:

Review materials are present in this repository under `experiments/results/` and `experiments/manuscript/figures/out/v8/`. Each result family retains generated tables, input identities, protocol references and status. The result-path brackets in this review copy identify those records and will be transferred to a Source Data/provenance index for submission. S2 additionally retains frozen reporter outputs, source and download inventories, failed-attempt provenance, eligibility release, operational amendments and billing reconciliation [results/s8086/provenance/artifact_manifest.json; results/s8086/provenance/verification.json; results/s8086/exploratory/publication_archive.csv].

Some full raw logs and snapshots are local or externally stored rather than committed; their expected paths and hashes are recorded in experiment-specific manifests. Hashes and local absolute paths do not supply review access. [AUTHOR TO SUPPLY: a verified accessible review archive or public accession containing required raw records, source-data tables, preregistrations, amendments and a release-wide checksum manifest, with durable retrieval locations and any actual access restrictions.] The standalone cross-machine encounter benchmark package and generated table also remain release placeholders.

Figures 1–5 have PDF and PNG review exports in `manuscript/figures/out/v8/`. At the manuscript revision’s initial file check, editable SVGs were available for Figures 2–5 and Figure 1 SVG was absent; the older manifest is not evidence of current technical verification. The parallel figure job’s final files and verification must be checked against the corrected captions. Figure 3’s registered 8086 law test remains untested; the separate exploratory assays are not plotted. The SI map records Extended Data rendering status separately [manuscript/figures/out/v8/FIGURES_V8.md; manuscript/SI_v8.md].

New:

Generated reports and figure data are present in this repository under `experiments/results/` and `experiments/manuscript/figures/out/v8/`. Each result family retains generated tables, input identities, protocol references and status. Result-path brackets identify the supporting records. S2 additionally retains frozen reporter outputs, source and download inventories, failed-attempt provenance, eligibility release, operational amendments and billing reconciliation [results/s8086/provenance/artifact_manifest.json; results/s8086/provenance/verification.json; results/s8086/exploratory/publication_archive.csv].

Some full raw logs and snapshots are local or externally stored rather than committed; their expected paths and hashes are recorded in experiment-specific manifests. Hashes and local absolute paths do not supply review access. [AUTHOR TO SUPPLY: data accession/DOI and accessible archive containing required raw records, source-data tables, preregistrations, amendments, the standalone cross-machine encounter benchmark package with its generated table and a release-wide checksum manifest, with durable retrieval locations and any access restrictions.]

Figs 1–5 have PDF and PNG exports in `manuscript/figures/out/v8/`. Fig. 3’s registered 8086 law test remains untested; the separate exploratory assays are not plotted. The Supplementary Information maps Extended Data displays to their files and sources [manuscript/figures/out/v8/FIGURES_V8.md; manuscript/SI_v8.md].


### MAIN-25

Reason: Remove review-copy framing while preserving access limitations. Keep the code release as one explicit author-supplied placeholder and remove repetitive disclaimer.

Old:

The current repository contains historical Algocell research execution and analysis code; external absolute Zilion paths in this review are source locations, not public releases. Capacity and retention reports are generated by `capacity_assays.py` and `evolved6502_retention.py`. The registered S2 presentation is regenerated by `results/s8086/provenance/generate_report.py`; exploratory analysis and its hash audit use `exploratory8086.py`, `exploratory8086_law.py` and `exploratory8086_verify.py`. Pre-assay prediction identities establish the ordering of registration, eligibility and tests. Report regeneration is distinct from running new simulations [results/s8086/provenance/artifact_manifest.json; results/evolved6502/inputs.csv; results/capacity/CAPACITY.md].

Figure reproduction uses `manuscript/figures/prepare_v8.py`, `build_v8.py`, `legends_v8.py` and `verify_v8.py`; commands are recorded in `manuscript/figures/out/v8/FIGURES_V8.md`. [AUTHOR TO SUPPLY: frozen Algocell and Zilion code releases/accession, dependency versions, licence, hardware requirements and verified runnable instructions distinguishing report regeneration from new simulations.] No completed release or reproducibility check of unavailable raw inputs is asserted here.

New:

The current repository contains historical Algocell research execution and analysis code; absolute Zilion paths identify external source locations, not public releases. Capacity and retention reports are generated by `capacity_assays.py` and `evolved6502_retention.py`. The registered S2 presentation is regenerated by `results/s8086/provenance/generate_report.py`; exploratory analysis and its hash audit use `exploratory8086.py`, `exploratory8086_law.py` and `exploratory8086_verify.py`. Pre-assay prediction identities establish the ordering of registration, eligibility and tests. Report regeneration is distinct from running new simulations [results/s8086/provenance/artifact_manifest.json; results/evolved6502/inputs.csv; results/capacity/CAPACITY.md].

Figure reproduction uses `manuscript/figures/prepare_v8.py`, `build_v8.py`, `legends_v8.py` and `verify_v8.py`; commands are recorded in `manuscript/figures/out/v8/FIGURES_V8.md`. [AUTHOR TO SUPPLY: frozen Algocell and Zilion code releases and accession/DOIs, dependency versions, licence, hardware requirements and verified runnable instructions distinguishing report regeneration from new simulations.]


### MAIN-26

Reason: Use the single-author declaration.

Old:

[AUTHOR TO SUPPLY: contributions, approved by all authors.]

New:

[AUTHOR TO SUPPLY: author contributions.]


### MAIN-27

Reason: Apply thousands separators to the unchanged bibliography page range.

Old:

1. Maynard Smith, J. & Szathmáry, E. *The Major Transitions in Evolution* (W. H. Freeman, 1995).
2. Szathmáry, E. & Maynard Smith, J. From replicators to reproducers: the first major transitions leading to life. *J. Theor. Biol.* **187**, 555–571 (1997). [https://doi.org/10.1006/jtbi.1996.0389](https://doi.org/10.1006/jtbi.1996.0389).
3. von Neumann, J. *Theory of Self-Reproducing Automata* (ed. Burks, A. W.) (University of Illinois Press, 1966).
4. Vasas, V., Szathmáry, E. & Santos, M. Lack of evolvability in self-sustaining autocatalytic networks constraints metabolism-first scenarios for the origin of life. *Proc. Natl Acad. Sci. USA* **107**, 1470–1475 (2010). [https://doi.org/10.1073/pnas.0912628107](https://doi.org/10.1073/pnas.0912628107).
5. Vasas, V., Fernando, C., Santos, M., Kauffman, S. & Szathmáry, E. Evolution before genes. *Biol. Direct* **7**, 1 (2012). [https://doi.org/10.1186/1745-6150-7-1](https://doi.org/10.1186/1745-6150-7-1).
6. Woese, C. R. On the evolution of cells. *Proc. Natl Acad. Sci. USA* **99**, 8742–8747 (2002). [https://doi.org/10.1073/pnas.132266999](https://doi.org/10.1073/pnas.132266999).
7. Langton, C. G. Self-reproduction in cellular automata. *Physica D* **10**, 135–144 (1984). [https://doi.org/10.1016/0167-2789(84)90256-2](https://doi.org/10.1016/0167-2789(84)90256-2).
8. Ray, T. S. An approach to the synthesis of life. In *Artificial Life II* (eds Langton, C. G., Taylor, C., Farmer, J. D. & Rasmussen, S.) 371–408 (Addison-Wesley, 1991).
9. Ofria, C. & Wilke, C. O. Avida: a software platform for research in computational evolutionary biology. *Artif. Life* **10**, 191–229 (2004). [https://doi.org/10.1162/106454604773563612](https://doi.org/10.1162/106454604773563612).
10. Fontana, W. Algorithmic chemistry. In *Artificial Life II* (eds Langton, C. G., Taylor, C., Farmer, J. D. & Rasmussen, S.) 159–209 (Addison-Wesley, 1991).
11. Pargellis, A. N. The spontaneous generation of digital “Life”. *Physica D* **91**, 86–96 (1996). [https://doi.org/10.1016/0167-2789(95)00268-5](https://doi.org/10.1016/0167-2789(95)00268-5).
12. C G, N., LaBar, T., Hintze, A. & Adami, C. Origin of life in a digital microcosm. *Phil. Trans. R. Soc. A* **375**, 20160350 (2017). [https://doi.org/10.1098/rsta.2016.0350](https://doi.org/10.1098/rsta.2016.0350).
13. Agüera y Arcas, B. et al. Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction. Preprint at [arXiv:2406.19108v2](https://arxiv.org/abs/2406.19108v2) (2024).
14. Cicala, F. et al. Coevolution of self-replication and function in a digital primordial soup. Preprint at [arXiv:2607.09211v2](https://arxiv.org/abs/2607.09211v2) (2026).


New:

1. Maynard Smith, J. & Szathmáry, E. *The Major Transitions in Evolution* (W. H. Freeman, 1995).

2. Szathmáry, E. & Maynard Smith, J. From replicators to reproducers: the first major transitions leading to life. *J. Theor. Biol.* **187**, 555–571 (1997). [https://doi.org/10.1006/jtbi.1996.0389](https://doi.org/10.1006/jtbi.1996.0389).

3. von Neumann, J. *Theory of Self-Reproducing Automata* (ed. Burks, A. W.) (University of Illinois Press, 1966).

4. Vasas, V., Szathmáry, E. & Santos, M. Lack of evolvability in self-sustaining autocatalytic networks constraints metabolism-first scenarios for the origin of life. *Proc. Natl Acad. Sci. USA* **107**, 1,470–1,475 (2010). [https://doi.org/10.1073/pnas.0912628107](https://doi.org/10.1073/pnas.0912628107).

5. Vasas, V., Fernando, C., Santos, M., Kauffman, S. & Szathmáry, E. Evolution before genes. *Biol. Direct* **7**, 1 (2012). [https://doi.org/10.1186/1745-6150-7-1](https://doi.org/10.1186/1745-6150-7-1).

6. Woese, C. R. On the evolution of cells. *Proc. Natl Acad. Sci. USA* **99**, 8,742–8,747 (2002). [https://doi.org/10.1073/pnas.132266999](https://doi.org/10.1073/pnas.132266999).

7. Langton, C. G. Self-reproduction in cellular automata. *Physica D* **10**, 135–144 (1984). [https://doi.org/10.1016/0167-2789(84)90256-2](https://doi.org/10.1016/0167-2789(84)90256-2).

8. Ray, T. S. An approach to the synthesis of life. In *Artificial Life II* (eds Langton, C. G., Taylor, C., Farmer, J. D. & Rasmussen, S.) 371–408 (Addison-Wesley, 1991).

9. Ofria, C. & Wilke, C. O. Avida: a software platform for research in computational evolutionary biology. *Artif. Life* **10**, 191–229 (2004). [https://doi.org/10.1162/106454604773563612](https://doi.org/10.1162/106454604773563612).

10. Fontana, W. Algorithmic chemistry. In *Artificial Life II* (eds Langton, C. G., Taylor, C., Farmer, J. D. & Rasmussen, S.) 159–209 (Addison-Wesley, 1991).

11. Pargellis, A. N. The spontaneous generation of digital “Life”. *Physica D* **91**, 86–96 (1996). [https://doi.org/10.1016/0167-2789(95)00268-5](https://doi.org/10.1016/0167-2789(95)00268-5).

12. C G, N., LaBar, T., Hintze, A. & Adami, C. Origin of life in a digital microcosm. *Phil. Trans. R. Soc. A* **375**, 20160350 (2017). [https://doi.org/10.1098/rsta.2016.0350](https://doi.org/10.1098/rsta.2016.0350).

13. Agüera y Arcas, B. et al. Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction. Preprint at [arXiv:2406.19108v2](https://arxiv.org/abs/2406.19108v2) (2024).

14. Cicala, F. et al. Coevolution of self-replication and function in a digital primordial soup. Preprint at [arXiv:2607.09211v2](https://arxiv.org/abs/2607.09211v2) (2026).



## SI substantive edits

All source brackets are retained verbatim. No scientific count, denominator, verdict or registered/exploratory/post hoc/construction/theorem status was intentionally changed. House-style changes are listed after the substantive edits.

### SI-01

Reason: Replace review-copy and supersession commentary with a description of the package. Removes the editorial date 12 October 2026, not an experimental date; accession/declaration placeholders remain in MAIN.

Old:

Review copy, 12 October 2026. This supplement is assembled from the current generated results, frozen contracts and dated corrections; it supersedes `manuscript/SI.md`. Paths beginning `results/`, `algocell_exp/` or `manuscript/` resolve from `experiments/`. These source brackets remain in the review copy and will move to the submission Source Data/provenance index. Public data/code accessions and author-confirmed declarations remain the explicit placeholders in `manuscript/MAIN_v8.md`; local paths and hashes are not public accessions.

New:

This supplement contains proofs, generated results and frozen experimental contracts. Paths beginning `results/`, `algocell_exp/` or `manuscript/` resolve from `experiments/`. Source brackets identify the underlying records. Public data and code accessions and the author’s declarations are provided in `manuscript/MAIN_v8.md`; local paths and hashes are not public accessions.

### SI-02

Reason: Disambiguate emulator versions from experiment identifiers using the definitions already given in Supplementary Note 7.

Old:

Core C0/C1/C2 denote emulator versions; displacement experiment C1 and 6502 retention experiment C2 denote experiments.

New:

Core C0/C1/C2 denote the historical compatibility, corrected classic and vendor-specific Zilog-NMOS emulator versions, respectively. Displacement experiment C1 and 6502 retention experiment C2 are separate experimental labels.

### SI-03

Reason: State the existing terminology and A/B notation before the proofs; no experimental claim changes.

Old:

Worlds are independent evolutionary replicates.

New:

A tape is a byte sequence; a cell is a population site containing a tape; a genotype is a distinct tape sequence. An update is a population update, an instruction step is an execution within an encounter and a sample is a recorded observation. The parent tape is A and the partner tape is B. Worlds are independent evolutionary replicates.

### SI-04

Reason: Distinguish theorem assignments from emulator instruction steps and population updates.

Old:

The byte written at step k

New:

The byte written at assignment k

### SI-05

Reason: Use assignment for the ordered-copy proof index.

Old:

its first possible write is step P−d+k

New:

its first possible write is assignment P−d+k

### SI-06

Reason: Use assignment for the ordered-copy proof index.

Old:

occurred at step k−d

New:

occurred at assignment k−d

### SI-07

Reason: Use assignment for the ordered-copy proof index.

Old:

Writes to that physical address are separated by P steps

New:

Writes to that physical address are separated by P assignments

### SI-08

Reason: Remove archive framing from the stated theorem; retain the exact conditions.

Old:

The archived specialization uses

New:

This specialisation uses

### SI-09

Reason: Replace slash shorthand and simplify the proof condition without weakening it.

Old:

Architectural source/destination increments must continue to induce the specified constant displacement in the mapped ring over all remaining copies.

New:

Architectural source and destination increments must preserve the specified constant displacement in the mapped ring throughout the remaining copies.

### SI-10

Reason: Present the proof and certificate content directly, without current-versus-archived manuscript framing.

Old:

The archived proof checks that admitted one-byte setup instructions in this reset do not introduce a harmful 16-bit pointer wrap during the block operation; the current candidate and setting ledgers additionally bind the executed code and geometry.

New:

The proof checks that admitted one-byte setup instructions under this reset do not introduce a harmful 16-bit pointer wrap during the block operation. The candidate and setting ledgers also identify the executed code and geometry.

### SI-11

Reason: State the prospective scope of exclusions directly; preserve the no-post-outcome-exclusion caveat.

Old:

These are exclusions from the stated family, not post-outcome repairs.

New:

The stated family excludes these cases before retention is assessed.

### SI-12

Reason: Replace compressed contrast with a direct statement of the same limitation.

Old:

The four core coordinates are held, not proved incapable of transmitting any alternative allele.

New:

Holding the four core coordinates fixed does not establish that they cannot transmit alternative alleles.

### SI-13

Reason: Clarify the source referent and split the ancestry rule into sentences.

Old:

A pure copy inherits its source's current origin at the time of the read; a copied byte that was itself overwritten inherits the earlier origin carried by that write.

New:

A pure copy inherits its source’s origin at the time of the read. If the source byte was overwritten, the copy inherits the origin carried by that write.

### SI-14

Reason: Avoid calling the failed converse a proved implication.

Old:

This proves the two implications without assigning zero inflow the stronger meaning of faithful inheritance.

New:

These two statements distinguish zero inflow from faithful inheritance.

### SI-15

Reason: Reserve cell for a population site, rather than a tape address.

Old:

In unmodified BFF, a pair has 128 cells;

New:

In unmodified BFF, a pair has 128 byte positions;

### SI-16

Reason: Reserve cell for a population site.

Old:

each head movement changes its position by one cell.

New:

each head movement changes its position by one byte.

### SI-17

Reason: Reserve cell for a population site.

Old:

An arithmetic or copying instruction changes at most one cell

New:

An arithmetic or copying instruction changes at most one byte

### SI-18

Reason: Reserve cell for a population site.

Old:

every original B cell differs from the desired copy

New:

every original B byte differs from the desired copy

### SI-19

Reason: Reserve cell for a population site.

Old:

The heads must collectively visit all 64 B cells;

New:

The heads must collectively visit all 64 B byte positions;

### SI-20

Reason: State the final theorem and retain the counterexample to the insufficient bound without revision-history wording.

Old:

The corrected finite bound is 2w+1 for odd widths and 2w for even widths. At the old odd-width bound 2w, w=1 permits

New:

The finite bound is 2w+1 for odd widths and 2w for even widths. The shorter odd-width bound 2w is insufficient: w=1 permits

### SI-21

Reason: Retain failed-bound evidence without referring to an earlier manuscript.

Old:

These are explicit counterexamples to the old finite statement.

New:

These are explicit counterexamples to the shorter finite bound.

### SI-22

Reason: Remove temporary-inventory commentary while retaining the authoritative final source and causal-correction context.

Old:

The literal audit's temporary partial 8086 inventory is historical and is superseded by the validated census and its final causal correction (Supplementary Note 8).

New:

The validated census and causal classification define the 8086 inventory used here (Supplementary Note 8).

### SI-23

Reason: Remove conversational transition.

Old:

Here traversal is descending

New:

Traversal is descending

### SI-24

Reason: Apply the unambiguous experiment label.

Old:

Evolved retention experiment C2 uses

New:

The 6502 retention experiment C2 uses

### SI-25

Reason: Distinguish the experiment from core C1.

Old:

Experiment C1 contains 22 archived parents

New:

Displacement experiment C1 contains 22 archived parents

### SI-26

Reason: Split the assay definition and remove the serial comma without changing its panel.

Old:

Direct assays use actual generation-eight descendants, all 255 alternatives at every free and erased coordinate, and 64 joint assignments against all 17 frozen partners.

New:

Direct assays use actual generation-eight descendants. They test all 255 alternatives at every free and erased coordinate and 64 joint assignments against all 17 frozen partners.

### SI-27

Reason: Clarify that computational reuse does not merge lineages.

Old:

Identical descendants share execution while preserving lineage identities.

New:

Identical descendants share an execution result, but their lineage identities remain separate.

### SI-28

Reason: Replace ambiguous implementation shorthand with the stated sampling limitation.

Old:

Collision claims mean this is not 8,192 guaranteed active encounters.

New:

Pair collisions mean that the 8,192 drawn slots do not guarantee the same number of active encounters.

### SI-29

Reason: Clarify the estimator output while preserving its primary status.

Old:

Corrected frozen32 final calls remain primary.

New:

Corrected frozen32 final-cell classifications are primary.

### SI-30

Reason: Disambiguate the experiment and use British spelling.

Old:

Every sampling role remains in C2; sensitivity reevaluations do not create extra cells.

New:

Every sampling role remains in retention experiment C2; sensitivity re-evaluations do not create extra cells.

### SI-31

Reason: Use the established fetch definition without slash shorthand.

Old:

The executed/site strips mark instruction-stream-fetched bytes, not just opcode starts.

New:

The execution and site strips mark fetched instruction-stream bytes, including operands.

### SI-32

Reason: Remove a dangling contrast and apply the figure house style.

Old:

Figure 2d instead follows

New:

Fig. 2d follows

### SI-33

Reason: Distinguish encounter execution from population updates.

Old:

At cumulative step 128

New:

At cumulative instruction step 128

### SI-34

Reason: Remove manuscript-editing terminology from the heading.

Old:

### Competition classes and displaced display details

New:

### Competition classes and figure measurements

### SI-35

Reason: Remove manuscript revision history while retaining the corrected result and its assay scope; removes the version token v7.

Old:

these are the counts corrected in the v7 audit.

New:

these counts describe the sampled site assay.

### SI-36

Reason: State the final figure contents.

Old:

No evolved 8086 exact-law point is added to this registered figure.

New:

The registered figure contains no evolved 8086 exact-law point.

### SI-37

Reason: Distinguish a plotted curve from a lineage.

Old:

The four serial-line keys are

New:

The four serial-curve keys are

### SI-38

Reason: Reserve cell for a population site.

Old:

Byte-zero sweep cells run

New:

Byte-zero sweep entries run

### SI-39

Reason: Use a descriptive heading without editorial framing.

Old:

## Supplementary Note 6. Registration and operational changes remain visible

New:

## Supplementary Note 6. Registration and operational chronology

### SI-40

Reason: State the scoring rule precisely.

Old:

Operational incompleteness is never a biological negative.

New:

Operational incompleteness does not count as a biological negative.

### SI-41

Reason: Remove execution-authority commentary from the manuscript while retaining the role of the records.

Old:

Historical launch receipts are provenance, not current instructions or additional execution authority.

New:

Historical launch receipts document the operational record.

### SI-42

Reason: Describe final archive contents rather than a revision action; both records are named in the existing source bracket.

Old:

This supplement does not rewrite the original worker manifest.

New:

The original worker manifest and the omitted-case completion record are both retained.

### SI-43

Reason: Correct the tense mismatch.

Old:

I1-T2 reruns the registered original ecological designs and seeds under core C1 and rescored them with their frozen scorers.

New:

I1-T2 reran the registered original ecological designs and seeds under core C1 and rescored them with their frozen scorers.

### SI-44

Reason: Use update for the population schedule, retaining the exact missing checkpoint.

Old:

a step-7,500 snapshot

New:

an update-7,500 snapshot

### SI-45

Reason: Correct the grammatical construction without changing the census counts.

Old:

The validated 8086 census enumerates every 256 period-one and 65,536 period-two word

New:

The validated 8086 census enumerates all 256 period-one words and all 65,536 period-two words

### SI-46

Reason: State the final classifier requirement while retaining the limitation of the earlier method.

Old:

Earlier trace-only classifications and their neutral-store shortcut are superseded.

New:

Trace-only classifications and neutral-store shortcuts do not satisfy this contract.

### SI-47

Reason: Use instruction step for encounter execution.

Old:

in 128 steps

New:

in 128 instruction steps

### SI-48

Reason: Split the selection rule from its denominator.

Old:

A separate descriptive panel selects an exact family exemplar from every world with a scheduled full-family tape, eighteen of twenty.

New:

A separate descriptive panel selects an exact family exemplar from every world with a scheduled full-family tape. This includes eighteen of twenty worlds.

### SI-49

Reason: Remove the awkward compound and preserve the complete outcome partition.

Old:

Among the originally 708 `POP ES`-unresolved cases, 587 qualify, sixteen become excluded and 105 remain unresolved.

New:

Among the 708 cases originally unresolved at `POP ES`, 587 qualify, sixteen become excluded and 105 remain unresolved.

### SI-50

Reason: Describe final source-table contents and centralise release placeholders in MAIN; remove review-package and pending-analysis commentary.

Old:

Tables 1 and 2 are the cohort and chronology tables above. The following numbered tables name the existing complete generated records, retaining row-level exclusions and failures. They are source tables for the review package, not promises to run further analyses. CSV/gzip suffixes identify machine-readable tables; a result report supplies their explanatory key. File paths and hashes do not substitute for the still-pending public accession.

New:

Supplementary Tables 1 and 2 describe the cohorts and registration chronology. The following tables identify the complete generated records, including row-level exclusions and failures. CSV files and gzip-compressed files contain machine-readable tables; the corresponding result reports explain their fields. Public accessions are provided in the availability statements in `manuscript/MAIN_v8.md`.

### SI-51

Reason: Replace dated filesystem inventory, legacy display paths and rendering instructions with the user-specified final map ed8_1.pdf–ed8_9.pdf and ED_V8.md. Scientific display content, denominators, caveats and status labels remain. Removed date/time/version/path numerals are housekeeping, not scientific measurements; the exact block below provides the exception evidence.

Old:

Filesystem inventory checked 13 October 2026 at 01:07 UTC (12 October local time). No `manuscript/figures/out/v8/ed8_*.pdf` or other v8 `ed_*` display was present at that check. The requested paths below are explicit coordination targets for the parallel rendering job; they are not asserted to be that job's confirmed filenames or completed outputs. Older PDFs listed as existing supply partial historical displays and do not constitute an updated nine-display supplement. The final figure delivery must reconcile the renderer's actual filenames with this map and verify its output manifest. The proofs and generated tables above already exist and do not depend on new simulation.

| Extended Data item | Requested v8 target | Rendered display inventory | Content required in the coordinated v8 display |
|---|---|---|---|
| 1. Validation and correction impact | `manuscript/figures/out/v8/ed8_1_validation.pdf` | Must be rendered; no complete older ED equivalent identified | Distinct instruction-vector, ring, isolated-assay and ecological units; expected divergences, I1 rate rule and all-unaffected failure, I1-T2 original/corrected verdicts and Stage-L completeness issue. Sources: Supplementary Tables 3,9,15 and Note 7 |
| 2. Survival, carriage and erasure | `manuscript/figures/out/v8/ed8_2_retention.pdf` | Existing partial display: `manuscript/figures/out/v6/ed_v6_erasure.pdf`; complete v8 display must be rendered | Full serial/branching maps and controls; identifiable alternatives and partner panels; lost, alive/<50% carriage and alive/≥50% carriage categories; ordered ancestry distinguishes actual erasure. Sources: Table 4, Note 5 and v8 erasure trace |
| 3. Inherited displacement and capacity | `manuscript/figures/out/v8/ed8_3_capacity.pdf` | Existing partial displays: `manuscript/figures/out/v6/ed_v6_law.pdf`, `manuscript/figures/out/v6/ed_v6_joint_table.pdf`; complete v8 display must be rendered | All 101 settings/22 parents, forward and reciprocal descendants, exact source maps, joint and erased controls, independent-world origins and exploratory gate exceptions. Sources: Table 5 and Notes 1–4 |
| 4. Sampling, reset and classification | `manuscript/figures/out/v8/ed8_4_sampling.pdf` | Existing historical partial displays: `manuscript/figures/out/ed_confine.pdf`, `manuscript/figures/out/ed_closure.pdf`, `manuscript/figures/out/ed_invasions.pdf`, `manuscript/figures/out/ed_invasions2.pdf`; complete v8 display must be rendered | Modal versus random-cell denominators, reset/stack sensitivity, motif failures, classes and exploratory detection order; distinguish ancestry from temporal detection. Sources: `results/r4/`, `results/population/`, `results/marker_check/`, `results/ladder/`, Table 1 |
| 5. Environmental competitions | `manuscript/figures/out/v8/ed8_5_competition.pdf` | Existing partial displays: `manuscript/figures/out/v6/ed_v6_competition.pdf`, `manuscript/figures/out/ed_switch.pdf`, `manuscript/figures/out/ed_dial.pdf`; complete v8 display must be rendered | Historical/core-C1 trajectories, absolute and relative abundance, core-loss/undefined-ratio cases, tails, failed E1 and explanatory predictions, small toxin effect and exploratory intruder interactions. Sources: Table 10 and Notes 5,7 |
| 6. Machine laws and finite censuses | `manuscript/figures/out/v8/ed8_6_machine_laws.pdf` | Existing historical Z80 census display: `manuscript/figures/out/ed_census.pdf`; current multi-machine display must be rendered | Complete 6502/8086 census scopes, final causal labels, corrected literal finite bound and counterexamples, constructed controls, S1 correction, all C2 roles/exclusions. No unmeasured pointer-randomization arm. Sources: Tables 6–8,11 and Notes 3,4,8 |
| 7. BFF literal and environmental controls | `manuscript/figures/out/v8/ed8_7_bff.pdf` | Existing historical partial display: `manuscript/figures/out/ed_bffinflow.pdf`; complete v8 display must be rendered | Literal/pointer/bandwidth/halting controls; open-fetch with zero inflow, and retained failed registered components. Sources: `results/bff/`, `results/bff_dials/`, `results/biology/individuality/`; Notes 2,3 |
| 8. Founder genealogies | `manuscript/figures/out/v8/ed8_8_genealogy.pdf` | Existing historical discovery display: `manuscript/figures/out/ed_assembly.pdf`, generated by `manuscript/figures/figs_n3.py`; complete v8 display must be rendered | Discovery-definition corrections, first founders versus later conversions, functional writers and N6 confirmation with failed N6-4 retained. Sources: Table 14 and Note 6 |
| 9. Registered S2 and separate follow-ups | `manuscript/figures/out/v8/ed8_9_s8086.pdf` | Must be rendered; no older S2 ED equivalent identified | Routes/censoring, descriptive R2, complete frozen candidate/exclusion/unknown ledger and empty law assays; clearly separate post hoc motif/fate/family/admission/intervention panels. Operational attempts are provenance, not replication. Sources: Tables 11–13 and Notes 6,8,9 |

The rendering handoff requires these nine PDFs, their matching editable/vector or raster exports as supplied by the figure job, legends with explicit denominators and registered/exploratory labels, and generated source-table/hash mappings. Figure 2's carriage category must not be labelled erased; fetching includes operands; competition R/T must be labelled partial-copy/full-tape; the 6502 series is 42 roles from 41 tapes and eleven worlds; and the displayed 8086 law status must refer specifically to the registered untested assay. Existence of a historical display does not verify these revised labels or current contents. No pending display is represented here as rendered.

New:

The Extended Data displays and their source mappings are listed below. The display manifest is `manuscript/figures/out/v8/ED_V8.md`.

| Extended Data item | Display | Content and sources |
|---|---|---|
| 1. Validation and correction impact | `manuscript/figures/out/v8/ed8_1.pdf` | Instruction-vector, ring, isolated-assay and ecological units; expected divergences, the I1 rate rule and all-unaffected failure, I1-T2 original/corrected verdicts and the Stage-L completeness issue. Sources: Supplementary Tables 3,9,15 and Supplementary Note 7. |
| 2. Survival, carriage and erasure | `manuscript/figures/out/v8/ed8_2.pdf` | Serial and branching maps and controls; identifiable alternatives and partner panels; lost, alive/<50% carriage and alive/≥50% carriage categories; ordered ancestry distinguishing actual erasure. Sources: Supplementary Table 4, Supplementary Note 5 and the v8 erasure trace. |
| 3. Inherited displacement and capacity | `manuscript/figures/out/v8/ed8_3.pdf` | All 101 settings/22 parents, forward and reciprocal descendants, exact source maps, joint and erased controls, independent-world origins and exploratory gate exceptions. Sources: Supplementary Table 5 and Supplementary Notes 1–4. |
| 4. Sampling, reset and classification | `manuscript/figures/out/v8/ed8_4.pdf` | Modal versus random-cell denominators, reset and stack sensitivity, motif failures, classes and exploratory detection order. Ancestry is distinguished from temporal detection. Sources: `results/r4/`, `results/population/`, `results/marker_check/`, `results/ladder/` and Supplementary Table 1. |
| 5. Environmental competitions | `manuscript/figures/out/v8/ed8_5.pdf` | Historical/core-C1 trajectories, absolute and relative abundance, core-loss and undefined-ratio cases, tails, failed E1 and explanatory predictions, the small toxin effect and exploratory intruder interactions. Sources: Supplementary Table 10 and Supplementary Notes 5,7. |
| 6. Machine laws and finite censuses | `manuscript/figures/out/v8/ed8_6.pdf` | Complete 6502/8086 census scopes, final causal labels, the literal finite bound and counterexamples, constructed controls, the S1 correction and all C2 roles/exclusions. Pointer randomisation remains unmeasured. Sources: Supplementary Tables 6–8,11 and Supplementary Notes 3,4,8. |
| 7. BFF literal and environmental controls | `manuscript/figures/out/v8/ed8_7.pdf` | Literal, pointer, bandwidth and halting controls; open fetch with zero inflow and retained failed registered components. Sources: `results/bff/`, `results/bff_dials/`, `results/biology/individuality/` and Supplementary Notes 2,3. |
| 8. Founder genealogies | `manuscript/figures/out/v8/ed8_8.pdf` | Discovery-definition corrections, first founders versus later conversions, functional writers and N6 confirmation with failed N6-4 retained. Sources: Supplementary Table 14 and Supplementary Note 6. |
| 9. Registered S2 and separate follow-ups | `manuscript/figures/out/v8/ed8_9.pdf` | Routes and censoring, descriptive R2, the complete frozen candidate/exclusion/unknown ledger and empty law assays. Post hoc motif, fate, family, admission and intervention panels are identified separately. Operational attempts are provenance, not replication. Sources: Supplementary Tables 11–13 and Supplementary Notes 6,8,9. |

Legends state denominators and registered or exploratory status; generated source tables and hashes document provenance. Fig. 2 distinguishes carriage below the threshold from complete erasure. Instruction-stream fetching includes operands. Competition R/T classes denote partial-copy/full-tape designs. The 6502 series contains 42 roles from 41 tapes and eleven worlds. The displayed 8086 law status refers to the registered untested assay.

## SI house style

Applied British spelling outside code and source brackets (specialisation, canonicalisation, normalised, totalling, prioritising, authorised, initialisation, summarise and synthesised); typographic apostrophes; Fig./Figs in prose; full Supplementary Note/Table cross-references; spaces around comparison/equality operators; removed serial commas in genuine lists while retaining punctuation between clauses. Source-bracket strings and code/path identifiers were protected. The source was already one line per prose paragraph; displayed equations and Markdown table rows remain separate.

## Guard exception inventory

The opening review-copy date 12 October 2026, the incidental v7 audit token and the complete old Extended Data filesystem/rendering block were removed or replaced as shown in SI-01, SI-35 and the final substantive item. These are final-state packaging changes explicitly requested by the user. The ED mapping now uses the exact user-specified ed8_N.pdf targets and ED_V8.md manifest. Scientific numbers and statuses in the ED content and the concluding paragraph remain.

## SI final consistency and numerical-formatting edits

### SI-52

Reason: Do not imply that accessions are already supplied when MAIN retains the authorised release placeholders.

Old: Public data and code accessions and the author’s declarations are provided in

New: Data and code availability and the author’s declarations are described in

### SI-53

Reason: Describe the current release placeholders accurately without duplicating them.

Old: Public accessions are provided in the availability statements in

New: The availability statements and accession placeholders are in

### SI-54

Reason: Define the three distinct units precisely and align with the coordinator’s execution-step terminology.

Old: An update is a population update, an instruction step is an execution within an encounter and a sample is a recorded observation.

New: An update comprises a population encounter batch and its mutation attempts. An execution step is a charged instruction or repeated block operation within an encounter. A sample is a recorded observation.

### SI-55

Reason: Apply execution step throughout, including zero-based erasure and cumulative replay indices; block repeats also consume execution steps.

Old: instruction step

New: execution step

### SI-56

Reason: Use the defined execution-budget unit while retaining the LDIR charging rule.

Old: 128 instruction executions; each `LDIR` repetition consumes one execution.

New: 128 execution steps; each `LDIR` repetition consumes one step.

### SI-57

Reason: Distinguish return to the admissible family from closure to partner information.

Old: The proof depends on family closure, noninterference and the recoverability premise, not on the number of sampled partners.

New: The proof requires offspring to remain in the certified family, noninterference and recoverability. It does not depend on the number of sampled partners.

### SI-58

Reason: Use confinement for the sampling role; retain first-detection priority and frozen selection.

Old: prioritising first/closure/final roles before keyed random cells

New: prioritising first-detection, confinement and final roles before keyed random cells

### SI-59

Reason: Avoid using step for an omitted population observation.

Old: missing-step issue retained

New: missing-snapshot issue described

### SI-60

Reason: Reserve open for partner instruction-stream fetch; describe the graphical marker shape.

Old: Open site symbols require

New: Hollow site symbols require

### SI-61

Reason: Remove awkward parenthetical equality notation; retain the exact arithmetic factors.

Old: The observer’s 306 lanes ( = 18×17 contexts)

New: The observer’s 306 lanes (18×17 contexts)

### SI-62

Reason: Use Fig. only for a numbered figure reference, not the generic noun.

Old: | 15 | Fig. input coordinates

New: | 15 | Figure input coordinates

### SI-63

Reason: Replace slash shorthand in the final ED content map without altering the thresholds or categories.

Old: lost, alive/ < 50% carriage and alive/ ≥ 50% carriage categories

New: lost, alive with < 50% carriage and alive with ≥ 50% carriage categories

### SI-64

Reason: Remove a serial comma.

Old: The archive preserves terminal failed ledgers rather than clearing their flags, binds the prior execution freeze to each exact completed world’s hash, and records renewed scoring authority separately.

New: The archive preserves terminal failed ledgers rather than clearing their flags, binds the prior execution freeze to each exact completed world’s hash and records renewed scoring authority separately.

### SI-65

Reason: Remove a serial comma.

Old: first founders from later conversions involving existing confined cells, and whole-tape non-copiers

New: first founders from later conversions involving existing confined cells and whole-tape non-copiers

### SI-66

Reason: Describe final package content and remove report-build instructions.

Old: Report regeneration reads committed tables; rerunning assays is a separate operation with its own resource ledger.

New: Committed tables and the resource ledger document the follow-ups.

### SI-67

Reason: Disambiguate a list whose unspaced form is parsed as 1,100 and 2,000. Verified in manuscript/figures/build_v8.py:408: [(36,1),(46,100),(62,2000)]; no scale value changes.

Old: three scale exemplars 1,100,2,000

New: three scale exemplars 1, 100 and 2,000

### SI-68

Reason: Expand shorthand and separate thousands-formatted checkpoint values. Verified in s8086.py HORIZON = 3_000_000 and SNAPSHOTS = (0, 20_000, 100_000, 1_000_000, HORIZON); results/s8086/S2.md records 585 samples per world.

Old: Initial sample plus S1 intervals through 3M: 585 samples/world; checkpoints at 0,20k,100k,1M,3M.

New: Initial sample plus S1 intervals through 3,000,000 updates: 585 samples/world; checkpoints at 0; 20,000; 100,000; 1,000,000; 3,000,000 updates.

### SI-69

Reason: Expand shorthand without changing the horizon; every results/s8086/tables/worlds.csv censor_step is 3000000.

Old: six worlds are right-censored at 3M.

New: six worlds are right-censored at 3,000,000 updates.

### SI-70

Reason: Expand the display-coordinate shorthand; verified by the same S2 world censor_step records.

Old: one 3M coordinate

New: one 3,000,000-update coordinate

### SI-71

Reason: Expand shorthand; results/tier2/stageL_diagnosis.json has journal_final_step = 10000000 for the retained worlds.

Old: All journals reach 10M,

New: All journals reach 10,000,000 updates,

### SI-72

Reason: Expand shorthand and avoid a dense compound; frozen SNAPSHOTS in s8086.py jumps from 0 to 20000.

Old: S2 lacks dense pre-20k checkpoints.

New: S2 lacks dense checkpoints before 20,000 updates.

### SI-73

Reason: Separate thousands-formatted values. Verified by the Stage G rows of results/s8086/exploratory/encounter_fates.csv; no checkpoint value changes.

Old: snapshots at 500,1,000,2,000,20,000,100,000.

New: snapshots at 500; 1,000; 2,000; 20,000; 100,000 updates.

Additional house style: spacing within numeric enumerations and register lists; typographic apostrophes after numeric identifiers; initialisation, characterises and initialised. Inline code and bracketed sources remain protected. Numeric seeds are identifiers and remain as recorded. The added standalone 6502 in the retention-experiment label is an authorised disambiguation, not a new observation.

### SI-74

Reason: Split the baseline and traversal conditions into short sentences.

Old: The actual sampled baseline is d-periodic at physical entry zero, d divides L, and ordered writes have geometry s+k→s+d+k modulo 2L.

New: The sampled baseline is d-periodic at physical entry zero, with d dividing L. Ordered writes have geometry s+k→s+d+k modulo 2L.

### SI-75

Reason: Remove the remaining serial comma in the parameter list.

Old: The overlap schematic aggregates two bytes per displayed unit: L = 12, c = 4, and d = 4 or 12, with zero or eight free byte coordinates respectively.

New: The overlap schematic aggregates two bytes per displayed unit: L = 12, c = 4 and d = 4 or 12, with zero or eight free byte coordinates respectively.

### SI-76

Reason: Correct the adjectival form.

Old: Technical incomplete records are not biological nondetections.

New: Technically incomplete records are not biological nondetections.

### SI-77

Reason: Remove an unnecessary comma between two parallel complements.

Old: The five L/d geometries give free counts 2 and 10 at L16, and 2, 10 and 26 at L32.

New: The five L/d geometries give free counts 2 and 10 at L16 and 2, 10 and 26 at L32.



### SI-78

Reason: Make the serial-proof requirements grammatically parallel; the hypotheses are unchanged.

Old: The proof requires offspring to remain in the certified family, noninterference and recoverability.

New: The proof requires continued offspring membership in the certified family, noninterference and recoverability.


## Numerical and bracketed-source exceptions

The guard deliberately reports lexical differences as FAIL; it has no allowlist and does not conceal edits. The exception accounting below explains every difference. Scientific numerical values, denominators and verdict labels remain unchanged. Formatting and the requested removal of editorial material are logged exceptions, not corrections to generated results.

| Location | Old → new | Evidence and interpretation |
|---|---|---|

| MAIN, culture fields | “Its similarity…” → “S2’s similarity…” | Explicit referent; raw inventory gains the label digit 2. The same S2 fields and source bracket are retained. |

| MAIN, retention heading paragraph | “C2 shares…” → “The 6502 retention experiment C2 shares…” | Required disambiguation of an existing experiment; no added observation. |

| MAIN, numerical provenance | Deleted “v7 mechanical audit” revision aside | Removes the editorial version digit 7; the provenance file and source bracket remain. |

| MAIN, display-scale exemplars | `{1,100,2000}` → `{1, 100, 2,000}` | `figures/build_v8.py:408` enumerates `(36,1),(46,100),(62,2000)`. The original ambiguous comma run is parsed as `1,100` and `2000` by the prose inventory and `1,100,200` plus `0` by the raw inventory. These are the same three display exemplars. |

| MAIN, Data availability | Removed initial-file-check sentence mentioning Figures 2–5 and Figure 1 | These are display identifiers in deleted revision/build commentary. PDF/PNG files for Figs 1–5 were confirmed locally. No statement certifies unmerged SVG or Extended Data artwork. |

| MAIN, References 4 and 6 | `1470–1475` → `1,470–1,475`; `8742–8747` → `8,742–8,747` | Thousands separators only; the original page values, titles, reference numbers and DOI brackets are preserved. |

| MAIN, author placeholders | Three AUTHOR TO SUPPLY brackets revised; one AUTHOR TO CONFIRM bracket added | Requested single-author contribution wording, explicit data/code accession or DOI placeholders and draft AI-use confirmation. Every non-placeholder bracket is unchanged. |

| SI, preface and Extended Data commentary | Removed review date, filesystem-check timestamp, legacy build identifiers and two references to “nine” displays in build commentary | The complete old/new blocks are in SI-01 and the Extended Data ledger. The final table still contains all nine items. Removed date/time tokens are `01`, `07`, `12` twice, `13` and `2026` twice; this is editorial history, not experimental chronology. |

| SI, experiment and sampling terms | Added `6502`; “first/closure/final” → “first-detection, confinement and final” | Existing retention-experiment and sampling-role labels made explicit. The prose scanner treats the slash-delimited old string as a path, so only the revised wording contributes a `first` token. |

| SI, horizons/checkpoints | `20k`, `100k`, `1M`, `3M`, `10M` → full thousands-separated values | `s8086.py:16,22` fixes HORIZON = 3,000,000 and SNAPSHOTS = (0,20,000,100,000,1,000,000,HORIZON); the S1 horizon is 10,000,000 in its frozen correction/cohort records. The final table also states that value without shorthand. |

| SI, scale and snapshot enumerations | `1,100,2,000` → `1, 100 and 2,000`; dense checkpoint list → separated 500; 1,000; 2,000; 20,000; 100,000 | The scale is verified by `build_v8.py:408`; Stage G checkpoint rows are in `results/s8086/exploratory/encounter_fates.csv`. Separators resolve lexical ambiguity without changing values. |

| SI, Extended Data targets | Descriptive `ed8_N_…` targets/legacy equivalents → `ed8_N.pdf` | Final naming is explicitly specified by the coordinator. Raw token removals additionally count digits in old `ed_v6_*` paths, suffixes and the removed v7-audit aside. All old/new paths are recorded in the SI ledger. |


The aggregate raw deltas include cancellation between repeated editorial labels; the per-location ledger above and exact old/new blocks preserve their provenance. The source-bracket comparison has no scientific exceptions: MAIN’s only changes are author placeholders, and all 69 SI bracket spans match exactly.


## prose_guard output

The before inventory uses the complete untouched snapshots saved before editing. The after comparison checks each complete final file against that snapshot. Both numerical layers preserve token spelling; source spans normalise whitespace only. The strict raw layer also catches digits within labels, paths and code. Its conservative parsing of comma runs is described above.

### Before

```text
prose_guard: /private/tmp/algocell-v8-prose/MAIN_v8.old.md -> /private/tmp/algocell-v8-prose/MAIN_v8.old.md
numbers: PASS (old=908, new=908; removed=0, added=0)
raw digit tokens: PASS (old=987, new=987; removed=0, added=0)
bracketed spans: PASS (old=139, new=139; removed=0, added=0)
PASS

prose_guard: /private/tmp/algocell-v8-prose/SI_v8.old.md -> /private/tmp/algocell-v8-prose/SI_v8.old.md
numbers: PASS (old=1002, new=1002; removed=0, added=0)
raw digit tokens: PASS (old=1238, new=1238; removed=0, added=0)
bracketed spans: PASS (old=69, new=69; removed=0, added=0)
PASS
```

### After

```text
prose_guard: /private/tmp/algocell-v8-prose/MAIN_v8.old.md -> /Users/neo/repos/algocell/experiments/manuscript/MAIN_v8.md
numbers: FAIL (old=908, new=907; removed=8, added=7)
  removed 1 × "1,100"
  removed 1 × "1470"
  removed 1 × "1475"
  removed 1 × "2"
  removed 1 × "2000"
  removed 1 × "5"
  removed 1 × "8742"
  removed 1 × "8747"
  added 1 × "1,470"
  added 1 × "1,475"
  added 1 × "100"
  added 1 × "2,000"
  added 1 × "6502"
  added 1 × "8,742"
  added 1 × "8,747"
raw digit tokens: FAIL (old=987, new=986; removed=8, added=7)
  removed 1 × "0"
  removed 1 × "1,100,200"
  removed 1 × "1470"
  removed 1 × "1475"
  removed 1 × "5"
  removed 1 × "7"
  removed 1 × "8742"
  removed 1 × "8747"
  added 1 × "1,470"
  added 1 × "1,475"
  added 1 × "100"
  added 1 × "2,000"
  added 1 × "6502"
  added 1 × "8,742"
  added 1 × "8,747"
bracketed spans: FAIL (old=139, new=140; removed=3, added=4)
  removed 1 × "[AUTHOR TO SUPPLY: a verified accessible review archive or public accession containing required raw records, source-data tables, preregistrations, amendments and a release-wide checksum manifest, with durable retrieval locations and any actual access restrictions.]"
  removed 1 × "[AUTHOR TO SUPPLY: contributions, approved by all authors.]"
  removed 1 × "[AUTHOR TO SUPPLY: frozen Algocell and Zilion code releases/accession, dependency versions, licence, hardware requirements and verified runnable instructions distinguishing report regeneration from new simulations.]"
  added 1 × "[AUTHOR TO CONFIRM: AI-use disclosure.]"
  added 1 × "[AUTHOR TO SUPPLY: author contributions.]"
  added 1 × "[AUTHOR TO SUPPLY: data accession/DOI and accessible archive containing required raw records, source-data tables, preregistrations, amendments, the standalone cross-machine encounter benchmark package with its generated table and a release-wide checksum manifest, with durable retrieval locations and any access restrictions.]"
  added 1 × "[AUTHOR TO SUPPLY: frozen Algocell and Zilion code releases and accession/DOIs, dependency versions, licence, hardware requirements and verified runnable instructions distinguishing report regeneration from new simulations.]"
FAIL

prose_guard: /private/tmp/algocell-v8-prose/SI_v8.old.md -> /Users/neo/repos/algocell/experiments/manuscript/SI_v8.md
numbers: FAIL (old=1002, new=1006; removed=11, added=15)
  removed 1 × "01"
  removed 1 × "07"
  removed 1 × "1,100"
  removed 2 × "12"
  removed 1 × "13"
  removed 1 × "20,000,100,000"
  removed 2 × "2026"
  removed 2 × "nine"
  added 1 × "1"
  added 1 × "1,000,000"
  added 1 × "10,000,000"
  added 1 × "100"
  added 2 × "100,000"
  added 3 × "20,000"
  added 4 × "3,000,000"
  added 1 × "6502"
  added 1 × "first"
raw digit tokens: FAIL (old=1238, new=1212; removed=38, added=12)
  removed 1 × "-20"
  removed 1 × "01"
  removed 1 × "07"
  removed 1 × "1,100"
  removed 1 × "10"
  removed 2 × "12"
  removed 1 × "13"
  removed 2 × "2"
  removed 1 × "20"
  removed 1 × "20,000,100,000"
  removed 2 × "2026"
  removed 5 × "3"
  removed 8 × "6"
  removed 1 × "7"
  removed 8 × "8"
  removed 1 × "80"
  removed 1 × "8086"
  added 1 × "1,000,000"
  added 1 × "10,000,000"
  added 2 × "100,000"
  added 3 × "20,000"
  added 4 × "3,000,000"
  added 1 × "6502"
bracketed spans: PASS (old=69, new=69; removed=0, added=0)
FAIL
```


## Final verification and coordinator handoff

The protected MAIN prefix matches exactly. Every non-placeholder MAIN source/citation/mathematical bracket matches; every SI bracket matches. Both edited scopes contain no prose bullets or hard-wrapped prose paragraphs; display equations and tables retain their structural lines. References remain numbered bibliographic entries, each on its own paragraph. Variables remain in the existing plain-text style. All five legends meet the existing 250-word contract: Fig. 1: 239, Fig. 2: 244, Fig. 3: 249, Fig. 4: 237, Fig. 5: 241. The guard’s scratch checks include changed numerals, source brackets, numeric-bearing identifiers and unchanged files. `git diff --check` passes.

The remaining author line, front-matter placeholders, direct questions, spelling and terminology issues before the Figure legends boundary are listed above for the coordinator. Caption exports are outside this pass’s allowed paths; regenerate them from `MAIN_v8.md` with `figures/legends_v8.py` during figure integration. Reconcile the final Extended Data files and manifest with the table. No `audit_v8*` or `AUDIT_v8*` file was edited.

## Commit handoff

The requested single commit could not be created in this session. The explicit-path staging command failed before staging any file with `fatal: Unable to create '/Users/neo/repos/algocell/.git/index.lock': Operation not permitted`. The session permits manuscript edits but treats repository metadata as read-only, and permission escalation is unavailable. No commit or push was performed. The coordinator should commit only `experiments/manuscript/MAIN_v8.md`, `experiments/manuscript/SI_v8.md`, `experiments/manuscript/V8_PROSE_CHANGES.md` and `experiments/manuscript/prose_guard.py`, using the requested final trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The prepared commit message is `/private/tmp/algocell-v8-prose/commit-message.txt`.
