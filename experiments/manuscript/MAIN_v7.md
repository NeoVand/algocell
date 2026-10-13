# Title candidates

Reproduction can erase its own variation

An inherited copying rule controls what survives reproduction

The copying mechanism sets the capacity for heredity

# Working title: What spontaneous replicators pass on

*Manuscript v7. Source brackets resolve from `experiments/`, unless an absolute path is given. “Core C1” denotes the corrected Z80 semantics; “displacement experiment C1” denotes the registered capacity study. Registered outcomes, exploratory/post hoc analyses, constructions and theorems retain separate status. Figures refer to `manuscript/figures/out/v8/`.*

**Genuinely pending release items:** [fill: immutable public data and code accession/DOI, release URL and frozen release-wide manifest catalogue]; [fill: standalone cross-machine encounter benchmark package and generated benchmark table]. The experiment-specific manifests, certificates, result tables and v8 figure inputs already exist. These release items supply no missing scientific outcome and must be completed before submission.

## Abstract

Darwinian evolution requires descendants to inherit differences on which selection acts [Maynard Smith & Szathmáry 1995; Szathmáry & Maynard Smith 1997]. Program soups reveal spontaneous replication, but its transmission of introduced variation requires separate measurement [Agüera y Arcas et al. 2024; Cicala et al. 2026]. Here we use Zilion, an execution substrate we introduce, to distinguish survival from inheritance. Branching populations of a representative first Z80 replicator survived eight generations while none of 512 tested mutation types met the registered carriage criterion [results/branching/SUPP_TABLE.md]. In a recurring forward-copy family, an inherited displacement controls free payload. Registered edits and evolved 6502 assays agreed exactly with the overlap prediction [results/capacity/CAPACITY.md; results/evolved6502/RETENTION.md]. The prospective 8086 route prediction failed: an exact pusher was available, yet 12 of 14 first detections were confined [results/s8086/tables/worlds.csv]. Its registered copying-law test remains untested because no evolved candidate was certified [results/s8086/tables/law_denominators.csv]. Exploratory, post hoc analysis nevertheless found independent evolution of the same overlap geometry and exact law agreement in 2,224,640 encounters [results/s8086/exploratory/law_assay_denominators.csv; results/s8086/exploratory/law_assay_summary.csv]. Availability therefore does not decide the route, while copying geometry explains transmission wherever its conditions hold. Closure to partner information and broad transmission of variation are distinct achievements, providing a deliberately strict experimental analogue of Woese’s Darwinian threshold [Woese 2002].

## Main text

Natural selection acts on differences, so a lineage can evolve only if reproduction preserves the differences that
arise within it. This requirement separates replication from heredity. Von Neumann's self-reproducing automaton
copies a description that it also interprets [von Neumann 1966], and the distinction between limited and unlimited
heredity asks how many alternatives a reproducing system can carry [Szathmáry & Maynard Smith 1997]. Some
compositional systems return towards built-in states and fail to preserve selectable variants [Vasas, Szathmáry &
Santos 2010]. Appropriately organized autocatalytic networks, by contrast, can support limited heredity and adaptation
[Vasas et al. 2012]. Woese placed the onset of Darwinian evolution proper at a threshold, where cellular organization
became integrated enough for vertical descent to dominate the horizontal exchange of an earlier communal regime
[Woese 2002].

Artificial chemistries of interacting programs have made the origin of replication itself observable. In soups of
random programs that execute on each other's memory, self-replicators arise without a fitness function, and
replicating architectures succeed one another as the soup evolves [Agüera y Arcas et al. 2024; Cicala et al. 2026].
These studies already trace where copied bytes come from, describe compact copying cores, and show that some cores
copy random tails faithfully and survive repeated rounds of mutation and replication [Agüera y Arcas et al. 2024;
Cicala et al. 2026].

What has not been established is which introduced differences a spontaneously evolved copier preserves across
generations, separately from the survival of its carriers, and what in the copying operation decides. Abundance
alone cannot answer this. A successful copier can restore its pattern so effectively that it erases most variants
introduced into it, and faithful copying of an unmodified parent says nothing about copying of its mutants. We answer by introducing a difference into a parent, letting reproduction proceed, and following the difference separately from the lineage. We built a substrate that traces these copying operations and permits comparisons across processors with different instruction sets.

Here we build that substrate and use it to show that, in soups of Z80 machine code, what reproduction preserves is a
measurable property of the copying mechanism. Branching populations of a representative first replicator persist while losing tested variants by the registered carriage criterion. In a recurring family of later copiers, a single geometric property of the copy decides which parental bytes
reach descendants, the parameter that sets this geometry is itself inherited, and the chemistry of the environment
shifts which designs prevail. Comparing processors separates how replication begins from what it transmits.

### Reproduction exposes transmission

Zilion separates a processor's instruction semantics from the world in which programs meet (Fig. 1a). An encounter
places two tapes of L bytes on a ring of 2L bytes, resets the processor to a declared state, executes a bounded number
of instructions starting in the first tape, and writes both tapes back. The soup supplies encounters between lattice
neighbours and background mutation. It has no fitness function and does not replace unsuccessful programs, so a
program reproduces only if its execution writes, into some partner, a tape that reproduces in turn [algocell_exp/machines/; /Users/neo/repos/zilion/docs/ARCHITECTURE.md].

A versioned semantic contract specifies everything such a claim depends on: address reduction on the ring, initial
registers and stack phase, the unit of execution and the treatment of undefined instructions. Cell boundaries,
execution time and reset state are supplied by the world, while the code that exploits them evolves. This matters
because an evolved copier can depend on a register value supplied at reset as strongly as on an instruction in its own
tape (Methods).

We validated the processor cores against per-instruction reference tests. These are emulator-generated vectors for
the Z80 and 6502 and hardware-recorded vectors for the 8088. The historical populations analysed here were produced
with Algocell kernels, and we identify their assays, fresh reruns with the corrected Z80 core (core C1) and current Zilion
implementations separately. The core-C1 reruns retained 19 of 22 claim verdicts and changed 4 of 36 registered components,
and we report historical failures and C1 outcomes side by side [results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv] (Fig. 1d; Methods).

Four measurements, kept separate throughout, turn a soup into an experiment. Emergence curves and a representative world show when reproduction becomes detectable (Fig. 1b,c); isolated assays then ask what it preserves (Fig. 2). A culture test asks whether a
tape's offspring itself reproduces. An execution trace asks whether execution stays within the replicator's own bytes.
Inflow, measured with the parent fixed, is the variation of offspring across partners: how much the partner
contributes. Transfer assays follow a deliberately introduced parental difference through repeated reproduction.

These properties dissociate in both directions. A program can read a partner's bytes without executing them, and it
can execute partner addresses after overwriting their contents, so none of the measurements stands in for another
(Methods).

### Lineages survive while alleles vanish

In all 80 historical worlds of the registered benign-chemistry cohort, the first detected replicator was open in the
sampled encounters: its execution ran on into the partner it was overwriting [results/inflow_insoup/INFLOW_INSOUP.md,
X2-2]. Its most common form tiles the tape with the word `01 c5` [results/stageG/stageG/stage_g_runs.csv], first described
by Agüera y Arcas and colleagues [Agüera y Arcas et al. 2024]. The tile decodes as `01 c5 01`, the instruction
`LD BC,$01C5`, followed by `c5`, `PUSH BC`, so the same byte values serve as the opcodes and as the literal data
pushed into the partner's memory.

To ask what such a replicator transmits, we carried out a registered branching-transfer experiment (X1). In it, every
surviving member of a population reproduced against two fresh partners per generation, so that loss of a mutation
could be distinguished from extinction of its lineage (Fig. 2a). Under uniform random partners, 48 of 64 unmutated
populations of the 16-byte pusher survived for eight generations. Among mutant populations, 38% were still alive at
generation eight, yet none of the 512 tested mutation types had a population that met the registered criterion for
carrying its mutation. The registered prediction that the open replicator would retain heritable variation (X1-2)
therefore failed, and we report it as failed [results/branching/branching_summary.csv, pusher16, arm U;
results/branching/SUPP_TABLE.md; REVISION_PREREG.md, X1]. The pusher rebuilds its form while losing these variants by the registered carriage criterion.

Three confined Z80 replicators gave contrasting results under the same protocol (Fig. 2a). Two regenerators, one
returning control to itself and one using a block copy, preserved every unmutated population and scored no
transmissible site, whereas an evolved block-copy transmitter carried nine scored sites through all eight generations
[results/branching/branching_summary.csv, ret16, ldir32, genome20_s6003]. A site scores when at least half of the tested identifiable alternative values at that position meet the registered retention criterion. This is demanding, and rare
variants can persist below it: the 64-byte pusher retained 5 of 2,048 tested mutant types with uniform partners, and
the block-copy regenerator retained a few working core variants [results/branching/SUPP_TABLE.md].

The contrast is therefore not between all and nothing. It is between copiers that preserve broad classes of
alternatives at specific positions and copiers that preserve almost none. The confined regenerators and transmitter differ in what reproduction preserves despite identical unmutated-population survival. Historical serial transfers and byte maps show the corresponding retention contrast (Fig. 2b,c), while an ordered trace follows an introduced byte through its overwrite and the descendant’s next reproduction (Fig. 2d). That raises the question of what, in the copying operation, decides which differences
survive.

### Overlap sets what is inherited

For the most tractable family of evolved copiers, the answer is geometric (Fig. 3a). Consider a copy that moves bytes
around the ring, each written a fixed displacement d ahead of where it was read. The first d reads obtain original
parental bytes. After that, the read pointer reaches bytes already written and copies copies, so the first d bytes
are propagated around the whole ring and every byte beyond them is overwritten before it can be read. A small
displacement therefore regenerates a short pattern regardless of what else the parent held, whereas a displacement
spanning the whole parent reproduces it entire.

With the copying core of c bytes held fixed inside the first d, and with the stated initialization, offspring
alignment and forward coverage, the number of independently free payload positions is r = d − c [theorem: manuscript/MAIN_nature.md, Proposition 8; Methods]. The proof
follows every byte of the final ring back to its source and holds for jointly arbitrary payload values. It counts free
payload only; working variants of the core itself are additional possibilities that it does not count.

The Z80 soups evolve exactly this machine. In the four-byte core `XX 5e ed b0`, the first byte executes as a harmless
instruction and is then read as data by `LD E,(HL)` to set the displacement, and `LDIR` performs the overlapping copy.
The variant `1e NN ed b0` carries the displacement as an immediate operand. Cicala and colleagues identified this
compact core and noted its partner offset [Cicala et al. 2026]. What the theorem adds is its hereditary consequence,
r = d − 4 [theorem: manuscript/MAIN_nature.md, Proposition 8]. The core occurs in 2,347 of 15,655 sampled confined heritable cells, so it is a recurring architecture
rather than the whole ecology [results/offset/CORE_FAMILIES.md].

In an exploratory regrouping (Fig. 3b) of the cells that carry this motif with a displacement dividing the tape length, sampled
site scores closely match the prediction. Of 2,343 such cells, 2,342 had exactly d − 4 scored unexecuted transmissible
positions. In all 53 cells whose total score exceeded d − 4, the additional sites lay at positions that are fetched as
part of the instruction stream. The single deficit preserves its predicted payload in targeted transfers but fails the
whole-tape similarity gate of the population assay [exploratory: results/crystal/cells.csv,
results/crystal/cell_exceptions.csv, results/crystal/offset_examples.csv; results/offset/offset_cells.csv]. Motif and
divisibility alone do not certify every hypothesis of the theorem, and these are sampled scores, not exhaustive
capacities (Methods).

The law also corrects a tempting intuition: what matters is the displacement of the copy, not the visible period of the
tape. One evolved tape repeats every four bytes, all of them core, yet carries four transmissible positions, because its
copy displacement is eight [exploratory: results/crystal/cell_exceptions.csv, K, L = 32, seed 5006]. Hereditary
capacity belongs to the family of sequences that a copying operation preserves, not to the appearance of any one
member.

The same operation separates two properties that are easily run together. Under the theorem's hypotheses, every byte of
the offspring descends from the parent's protected prefix, whatever the partner initially held. Such an overlapping
regenerator can therefore exclude partner information entirely while erasing variation outside its protected prefix.
Closure removes an external source of difference, and transmission preserves an internal one.

### The copying rule is inherited

If the displacement is encoded in the replicator's own bytes, it should be copied with the core, and a change in it
should be inherited. Archived serial-transfer interventions on an evolved regenerator, ldir32, show that it is. When its displacement byte was changed from `04` to `08` or to `60`, the mutant markers persisted through eight
transfers [results/serial_retention/sr_mutants.csv.gz, ldir32, gen 8, pos 0, val 8 and 96]. Under the theorem these changes predict more free payload; the registered displacement experiment C1 measured that capacity directly (Fig. 3b,c). From 22 archived parents we made 101 frozen displacement settings, including original-value controls, with forward and reciprocal passages. Eligibility and predicted coordinates were fixed from code and traces before retention outcomes. In 16 lineages per setting and direction, followed for eight generations, the edited displacement and whole core were inherited. Every tested alternative at each predicted free coordinate in actual generation-eight descendants was transmitted; sampled joint assignments were transmitted and erased-coordinate controls lost their original ancestry. Across 1,067,608 direct cases there was no mismatch, with fixed-core capacities from 0 to 60 positions [registered: results/capacity/edit_eligibility.csv; results/capacity/serial.csv.gz; results/capacity/capacity_cases.csv.gz; results/capacity/CAPACITY.md]. Thus an inherited change to a copier with no free payload can permit most of its tape to vary, and a reciprocal edit restores the original erasure. This C1 experiment is distinct from the core-C1 correction study.

An exploratory scan (Fig. 3d) of all 256 values of this byte shows how wide the dial is [exploratory:
results/crystal/alleles_ldir32.csv]. Ninety-nine values give a working replicator, and 50 of these match the edited value at the aligned position in at least half of first-generation offspring; equality alone is not a serial inheritance certificate. The 11 values that pass the sampled eligibility
checks span displacements of 4 to 32 bytes and predicted payloads of 0 to 28 positions. The dial is specific to this
family. In the same exploratory scan, an evolved transmitter that copies backwards with `LDDR` had only 15 working
values, and it lies outside the forward theorem [results/crystal/alleles_genome20_s6003.csv].

This is an evolved instance of an old architectural idea. Von Neumann's automaton requires a description that is both
used and copied [von Neumann 1966]. The displacement byte is used to construct the copy and copied along with it.
Within this copying family it is a heritable trait whose predicted phenotype is the capacity to transmit arbitrary
payload. We do not claim that selection favours it for that reason: whether a larger payload benefits a lineage
depends on the environment.

### Environments sort hereditary designs

Without mutation, mixtures of regenerators and transmitters drift, with no consistent winner
[results/tier2/scored/E/REPORT.md, mutation off]. With mutation, chemistry shifts their relative abundance. In the core-C1 reruns of the
registered competitions at tape length 32, started with equal shares, median transmitter shares among classified regenerators and transmitters, T/(R+T), were 0.05 where a zero opcode executes harmlessly and 0.76 where fetching a zero opcode halts execution. All ten
lethal worlds exceeded one half, meeting the registered endpoint [results/tier2/scored/E/REPORT.md, L = 32,
mutation on; results/tier2/T2_COMPONENTS.csv, E-L32L] (Fig. 4a,b).

Both invasion directions ended at similar environment-specific medians: 0.09 and 0.08 under benign tar, and 0.81 and
0.77 under lethal tar, with five worlds per direction. These finite-horizon endpoints are consistent with convergence from either rare starting design, but do not establish equilibrium [results/tier2/scored/E/REPORT.md, L = 32]. In the core-C1 lethality dial, transmitters formed
a majority of heritable cells in 8 of 10 fully lethal worlds, against 6 of 10 historically, when this registered endpoint failed [results/tier2/T2_DZ.csv; REVISION_PREREG.md, DZ3].

The lethality dial also changes observed route frequencies and confinement times (Fig. 4c), although it does not impose a fixed sequence. Its corrected-core timing reversal is retained (Methods). Under benign chemistry, short-tape worlds often remain dominated by regenerators: they were the final
majority in 17 of 20 worlds at L = 16 and 14 of 20 at L = 20, whereas at L = 50 every world remained dominated by open
cells [results/ladder/population_worlds.csv]. In an exploratory analysis, regenerators were detected before
transmitters in 30 of the 42 benign worlds that contained both, an order of detection rather than of ancestry
[exploratory: results/ladder/LADDER.md]. Under lethal chemistry the picture differs. In an exploratory assay of the
separate historical lethal-tar cohort, all ten first detected tapes were confined, had zero sampled partner inflow and
carried at most four scored sites, yet by the final sample transmitters were a majority of heritable cells in nine of
the ten worlds [exploratory: results/crystal/first_cells.csv, Z80 I; results/ladder/population_worlds.csv, stage I].

Why lethal chemistry favours transmitters remains open. Registered tests rejected a toxin mechanism
[results/toxin/CAUSAL.md], backup cores [results/n2/BACKUP.md] and isolated mutational flows
[results/offset/OFFSET_FLOW.md; REVISION_PREREG.md, N1, N2 and E-mech]. Exploratory encounter assays locate an
asymmetry in how the two designs fare against core-free intruders, but they do not yet explain the balance
[exploratory: results/offset/INTRUDER.md]. Because a tape is both a copier and material that other programs execute
and overwrite, what a replicator preserves changes what its neighbours encounter. A mechanism for the balance will have
to predict how the balance moves after an intervention.

### Availability does not decide the route into copying

The comparison across processors separates an available copying mechanism from the route evolution takes (Fig. 5a–c). The 6502’s accumulator load–push channel is one byte wide. Our retrospectively derived theorem excludes an infinite periodic self-writer in this single-register grammar, not every finite encounter or every instruction family [theorem: results/literal_channel/CHECK.md]. Before the soups, a validated census found no nonzero exact zero-partner self-writer of period one to four at L = 16 under the declared 128- and 512-step arms [results/census6502/table_main.csv]. In registered S1 primary arm A, all 20 first detected replicators were confined; median detection was 1,125,000 updates. Only 14 met the additional registered loop criterion, so P1 was not supported. The ring-position explanation of the others is post hoc [results/s6502/v2/arm_comparison.csv; results/s6502/CORRECTION.md].

The copying law nevertheless transferred. Constructed 6502 controls gave the predicted sampled carriage in 296 qualifying cases, distinct from exact capacities. The complete registered C2 panel sampled 942 roles from all 80 S1 worlds and admitted 42 roles by frozen code-and-provenance rules. Their constrained footprints spanned 6–10 bytes and predicted payloads 0–9 positions. All predictions held: 91,428 direct interventions and 731,424 serial observations through generation eight gave no mismatch (Fig. 3b) [construction: results/census6502/table_closers_law.csv; registered: results/evolved6502/eligibility.csv; results/evolved6502/geometry.csv; results/evolved6502/law_verdict.csv; results/evolved6502/exact_cases.csv.gz; results/evolved6502/exact_serial.csv.gz]. Roles and interventions are nested observations, not independent worlds.

The coherent-fetch 8086 supplied the prospective challenge. Its exact pusher was available: the tile `B9 51` decodes as `MOV CX,$B951; PUSH CX` under the frozen reset [results/census8086/table_hits.csv; algocell_exp/machines/worlds_8086.json]. Yet its first detected replicators were predominantly confined (Fig. 5a,c). Registered S2 found copiers in 14 of 20 worlds, meeting R0, but only 2 of those 14 began open, failing R1; the 12 confined beginnings triggered the route kill rule. The census-motivated open-beginning claim is withdrawn. Detection had a Kaplan–Meier median of 652,500 updates, with six worlds censored at the fixed horizon (Fig. 5b). The timing comparison is descriptive [registered: results/s8086/tables/worlds.csv; results/s8086/tables/kaplan_meier.csv; results/s8086/S2.md]. Availability had supplied a possible route, not its probability of arrival or establishment.

The registered law test did not resolve the question. L1 found no eligible evolved member of its narrow REP family and is untested. Amended L2 also admitted none: 724 certificates remained unresolved, 708 at unmodelled `POP ES`. L2 is untested with support suspended, not a law failure or a pass with no mismatches. R2 mechanism occurrence and persistence were demoted to descriptive status before population data (Methods) [registered: results/s8086/tables/law_denominators.csv; results/s8086/tables/law_candidates.csv; results/s8086/S2.md].

Exploratory, post hoc follow-ups explain what the frozen extractor missed. No full `B9 51` tile in either phase, or alternating run of at least four bytes, was observed in 11,700 scheduled top-three records or 2,000,000 checkpoint cells. Arrival or establishment is therefore the bottleneck suggested by the archive; its historical cause is not identifiable. Planted-pusher encounters show phase- and role-dependent vulnerability, not a measured population growth rate [exploratory, post hoc: results/s8086/exploratory/motif_scan.csv.gz; results/s8086/exploratory/encounter_fates.csv].

The dominant first-copier family instead evolved the same regenerating overlap geometry as the Z80 on a different instruction set. Ten of the 14 first copiers tile `5F A4 x y` (Fig. 5d). Post hoc tracing identifies `POP DI; MOVSB; branch`: the stack read initializes displacement d = 4, and subsequent copies propagate the first four parental bytes. With this four-byte footprint fixed, free payload is zero, yet alternative branch bytes are transmitted through eight passages. Zero free payload is again not zero heritable variation [exploratory, post hoc: results/s8086/exploratory/first_family.csv; results/s8086/exploratory/first_family_copies.csv; results/s8086/exploratory/family_transmission.csv].

Extending the dependency verifier post hoc admitted 600 genotype-world candidates: 590 full-tape d = 16 copiers and 10 d = 4 regenerators. Predictions were then committed before assays on a purposive panel of 32 genotypes from 20 worlds. Every selected case agreed: no mismatch in 2,224,640 encounters. The full-tape cases preserve arbitrary tested payload under held footprints; they are not smaller-seed regenerators. These results support transfer of the geometric explanation, while leaving registered L1 and L2 untested [exploratory, post hoc: results/s8086/exploratory/eligibility_counts.csv; results/s8086/exploratory/law_panel_predictions.csv; results/s8086/exploratory/law_assay_denominators.csv; results/s8086/exploratory/law_assay_summary.csv].

### Closure and heredity are distinct

In a recurring family of spontaneous copiers, overlap determines which parental bytes reach descendants. An inherited displacement therefore controls free payload, and chemistry shifts the prevalence of the resulting designs. The processor comparison sharpens this claim. The prospective 8086 route prediction failed despite an available exact pusher, whereas exploratory analysis recovered the same overlap geometry and exact post hoc agreement. Availability does not decide the route; the copying law transfers when its geometric and execution conditions are met. A successful explanation of transmission need not be a successful prediction of evolutionary arrival.

The results separate closure to partner information from broad transmission of parental variation. With parent and encounter contract fixed, closure means that offspring no longer depend on the partner’s initial contents. A qualifying regenerator can achieve this while preserving few tested alternatives, including working core alleles. Transmitters preserve broad classes of variants at specific sites. Zero partner inflow is our deliberately strict analogue of Woese’s transition towards stable vertical descent; Woese did not require horizontal transfer to disappear [Woese 2002]. Confinement alone does not establish this closure. Within these limits, the Darwinian-threshold framing has content: stable, genealogically continuous reproduction can coexist with limited transmission of the variation on which selection could act.

The same distinction matters wherever replication is studied without genes. Recovery of a sequence or composition could reflect transmission, reconstruction or recapture from the environment. Following an introduced difference, the survival of its carriers and the source of recovered material distinguishes these possibilities. We do not claim that chemistry uses a displacement byte. This is an inspectable case in which the question has an exact answer: which alternatives does reproduction preserve, and which does its own operation remove?

The remaining limits are substantive. Exact registered support comes from edited Z80 descendants and the eligible evolved 6502 panel; the 8086 follow-ups are exploratory and require a new prospective test with adequate extraction coverage. Finite joint challenges do not enumerate every possible payload. Nor has useful payload function or selection for increased capacity been demonstrated. An inherited difference must alter performance in a specified environment to establish adaptive use. Material that is both copied and interpreted, as in von Neumann’s description, offers one route; whether such use evolves in these soups without an imposed task remains open [von Neumann 1966]. Zilion makes these questions testable across explicitly specified processor contracts: not only whether a program reproduces, but which differences its reproduction preserves.

## Figure legends

**Figure 1. Reproduction made inspectable.**

**a,** Conceptual encounter panel. Two tapes occupy one encounter ring. The first Z80 pusher uses the same
byte values as instructions and literal output: `01 c5 01` loads BC and the next `c5`
pushes it. Writes begin at the partner's far end and move backwards while execution
runs into the partner. The return closer uses `c0` (`RET NZ`) to bring execution back
inside its own tape; it still writes into occupied partner memory. The vermilion return
instruction and arc are retained. Ellipses abbreviate tapes; this is a conceptual illustration, not a complete executor trace. The designs are separate observations, not a documented ancestor–descendant pair; `e3` exchanges HL with stack memory. [results/concept/traces.json; algocell_exp/shader/; manuscript/figures/out/v8/LEGENDS_V8.md]

**b,** Kaplan-Meier cumulative emergence of a heritable replicator in the historical
`none@nominal`, 128-instruction, four-neighbour Stage E cohort. Each plotted length has
10 independent worlds; nondetections are censored
at 300,000 steps. The curve is one minus Kaplan-Meier survival, using
the same calculation and length palette as `make_figures.py::km_panel`. The log axis
begins at 40 steps, matching the previous figure. Source: `results/stageE/assays.csv`;
generated world records and coordinates: `emergence_worlds.csv`, `emergence_curves.csv`. [manuscript/figures/out/v8/data/emergence_worlds.csv; manuscript/figures/out/v8/data/emergence_curves.csv]

**c,** One historical illustrative world, seed 2002, with 20,000
cells and 16 bytes per tape. Modal match is the fraction of cells matching
at least 12 of 16 bytes of the contemporaneous modal tape. Zero bytes are divided by
all bytes; distinct tapes are divided by all cells. This sequence-similarity neighbourhood
is not a spatial neighbourhood or a median world. The reference copiers assayed in
Figure 2 are not asserted to descend from this illustrative world. Source: archived
`video_L16_st128_k4_s2002.jsonl`, prepared by `prepare_v6.py`, reused as `world_trajectory.csv`. [manuscript/figures/out/v8/data/world_trajectory.csv]

**d,** Five separate validation units. Architectural raw matches are
3,126,917/3,137,000
hardware-recorded 8088 instruction vectors. Their 10,083
mismatches are expected divergences; there are zero unexpected architectural divergences.
Queue timing lies outside the vector contract. The other rows report core-C1 audit
outcomes: 643/690
ring conditions with a 95% upper bound on changed final-pair-memory rate below 0.01%,
407/417
isolated assay combinations, 19/22
registered claim verdicts, and 32/36
claim components. A whole claim combines its registered pass/fail tests; a component is
one constituent test. Grey means hardware state agreement, ring conditions classified
unaffected under the registered rate bound, identical isolated-assay metrics, or unchanged
pass/fail verdicts, according to row. Ochre marks the expected 8088 hardware divergences;
vermilion marks flagged ring conditions or changed assay metrics/verdicts.
The rows have distinct denominators and are not a pooled accuracy measure. Architectural
validation does not establish an 8086 ecological result. Sources: `results/impact/outcome.csv`,
`results/tier2/T2_CLAIMS.csv`, `T2_COMPONENTS.csv`, and the generated architectural baseline,
prepared as `validation_fractions.csv` by `prepare_v62.py`. [manuscript/figures/out/v8/data/validation_fractions.csv; manuscript/figures/out/v8/data/methods.csv]

**Figure 2. Survival is not retention.**

**a,** Generation-eight outcome partition of registered branching populations with uniform
random partners: lost lineage, surviving population with its marker erased, or surviving
population carrying the introduced marker. The entire starting mutant denominator is used,
not just surviving populations. Each site has 32 tested alternative byte values, with four
independently founded populations per alternative. Each surviving member reproduces against
two fresh partners per generation, with the registered population cap. The side columns give
unmutated control survival out of 64 populations and scored sites out of tape length.
The historical and core-C1 endpoints agree exactly in every plotted outcome and side-column count;
they are therefore represented by one bar, explicitly labelled. pusher: 48/64 unmutated populations alive; 0.3828 of mutant starts alive; 0.0000 carrying; 0/16 scored sites; return closer: 64/64 unmutated populations alive; 0.6074 of mutant starts alive; 0.0000 carrying; 0/16 scored sites; block-copy regenerator: 64/64 unmutated populations alive; 0.8787 of mutant starts alive; 0.0029 carrying; 0/32 scored sites; transmitter: 64/64 unmutated populations alive; 0.7535 of mutant starts alive; 0.4703 carrying; 9/20 scored sites.
Pusher X1-2 is a completed experiment whose prediction failed. A population carries when at
least half its members contain the mutant's centred five-byte window after cyclic alignment;
site scoring additionally uses the registered background correction and at least half the
tested alternatives. Rare carriers can exist below the site threshold, as the narrow teal
segment for the block-copy regenerator shows. Source: `results/branching/branching_summary.csv`
and `results/tier2/scored/Branching/branching_summary.csv`, reused as `branching.csv` and
`branching_endpoints.csv`. [manuscript/figures/out/v8/data/branching.csv; manuscript/figures/out/v8/data/branching_endpoints.csv]

**b,** Historical serial-transfer retention for the same four designs as panel a,
at transfers 1, 2, 4 and 8. Each line follows one explicitly named design, using distinct colours for the named designs. The denominator is all mutant serial starts,
not surviving starts. These serial
assays use all 255 alternative values and 16 serial lineages per alternative. They are
distinct from panel a's branching assay and are not presented as core-C1 reruns. Every curve
stays at its true transfer and measured fraction, including coincident low-retention
curves. Source: `results/serial_retention/sr_summary.csv`, generated as `serial_summary.csv`;
the presentation derives from `_sr_curves` in `make_figures.py`. [manuscript/figures/out/v8/data/serial_summary.csv]

**c,** Per-byte executed/site strips for six historical designs, from the archived serial assays. Grey cells were fetched during the reference encounters, white cells
were not. Open dark-teal circles identify thresholded sites carried at transfer one but
not eight; filled circles identify sites carried through transfer eight. All strips have
equal display width and different tape lengths, stated in each row; cell widths therefore
differ. Numeric columns report fetched bytes, sites at transfers one and eight, and the
number of transfer-eight sites never fetched. The evolved 20-byte row here is world 4
(`genome20_s6004`); panel a's branching transmitter is world 3 (`genome20_s6003`), and their
site counts must not be equated. The 50-byte pusher uses 32 seeded alternatives per site;
the other displayed tapes use all 255. Each alternative has 16 serial starts. A site requires at least half of identifiable alternative
values to meet the background-corrected retention criterion. Sources: `serial_summary.csv`
and `serial_sites.csv`, generated from the committed historical serial-retention result tables. [manuscript/figures/out/v8/data/serial_summary.csv; manuscript/figures/out/v8/data/serial_sites.csv]

**d,** One exploratory `ldir32` regenerator lineage, with a deliberately introduced mutation
at position 8, against zero partners through two complete 128-instruction encounters. Every
write is plotted in executor order. The vermilion circle identifies its overwrite at step 6,
when a byte descended from original position 0 is read at address 4 and writes hexadecimal 04 over marker A7 at address 8. The adjoining ancestry
trace counts ring bytes causally descended from the introduced marker, using all ordered
events. Its count drops from one to zero at that write and stays zero in the descendant encounter.
Only the initially edited byte receives the ancestry label; this differs from panel b's
five-byte-window carriage score. Addresses 0-31 are the focal parent and 32-63 its partner,
with modulo-64 wrapping. The circled write carries ancestry from original position 0 to destination 8 and removes the last causal descendant of the edit, not merely its byte value.
At cumulative step 128, the actual descendant becomes the focal parent and processor state
resets. The original marker has no remaining causal descendants and stays absent through
descendant reproduction. This is one illustrative lineage, not an estimate of population
survival. Sources: `erasure_ordered.csv`, `erasure_manifest.json`, generated by the committed
`archive_erasure_v6.py` script. Full memory/ancestry states remain in `out/v6/data/erasure_states.csv`. [manuscript/figures/out/v8/data/erasure_ordered.csv; manuscript/figures/out/v8/data/erasure_manifest.json]

**Figure 3. An inherited copying rule controls free payload.**

**a,** Abstract forward-overlap provenance drawn with `concept.py` cells, arrows and brackets.
Each displayed unit aggregates two bytes; both tapes have L=12, with a
c=4-byte copying core, K0/K1. The common before-copy track shows parent A
and occupied partner B. Arrows locate the initial read and the two possible initial writes;
each branch shows offspring B after copying. At d=4, the read head reaches
its own earlier writes, so copies of the core fill both tapes and free payload is zero.
At d=L=12, all original parent positions are read before overwrite, and
the four independent payload units represent 8 free byte
positions. Repeated K0/K1 labels identify repeated
descendants, not independent source positions. Under the stated protected prefix, initial
phase, offspring alignment and coverage, r=d-c. The units and ancestry are generated in
`geometry_units.csv` by `prepare_v8.py`. The schematic shows the theorem’s ancestry under its stated conditions. [theorem; manuscript/figures/out/v8/data/geometry_units.csv]

**b,** Prediction versus measured free positions, with an identity line and an unshifted
enlargement of 0-10 positions. Grey discs show exploratory sampled unexecuted-site scores
from 2,343 Z80 population cells; their areas are exactly proportional
to cell count, with the same scale in both views and in the three-disc key. Small population
discs are drawn over larger ones. Teal rings show exact capacities for 101 registered C1 displacement settings, including original-value controls, in 22 archived Z80 parents. Ochre triangles show
42 eligible evolved 6502 C2 roles; open charcoal squares show sampled
aligned carriage for 296 qualifying constructed 6502 control cases.
The latter are not exhaustive capacity measurements. All series remain at their true
coordinates; concentric symbols preserve coincidence. Exact agreements lie on y=x.
The vermilion ring marks the population gate deficit at predicted 4, sampled site count 0.
The whole-tape gate admits only offspring with at least 75% best-shift match to the mutant
parent before scoring its aligned alternative-byte carriage;
targeted transfers preserve its predicted payload, so this is a similarity-gate deficit,
not evidence of zero copying capacity. Population scores are exploratory cells nested
within worlds, not independent replicates. Motif/divisibility screening alone does not
certify every theorem prerequisite. The registered evolved 8086 law test is untested: the frozen extractor certified no eligible copier.
S2 L1 has zero eligible cases; amended L2 has unresolved certificates and support suspended.
No evolved 8086 law measurement is plotted. The later exploratory, post hoc agreement is reported separately in the text and does not change this registered verdict. Constructed controls do not supply evolved support. Sources: `law_population.csv`, `law_z80_c1.csv`,
`law_6502_c2.csv`, `law_6502_controls.csv`, generated by `prepare_v62.py` from the offset,
crystal, registered capacity, evolved6502 and census6502 result tables. [manuscript/figures/out/v8/data/law_population.csv; manuscript/figures/out/v8/data/law_z80_c1.csv; manuscript/figures/out/v8/data/law_6502_c2.csv; manuscript/figures/out/v8/data/law_6502_controls.csv; results/s8086/tables/law_denominators.csv]

**c,** Two parallel registered C1 intervention lineages, both shown at generation eight
and with the same 32 offspring coordinates. The original byte 04 gives displacement 4:
four core positions are kept while all 28 noncore parental positions are erased. Editing byte 0
from hexadecimal 04 to 60 gives displacement 32 and transmits all 28 payload positions.
Grey marks core, vermilion erasure and teal transmission. Exact causal-source indices,
not superficial equality, define the strip states. The edit is inherited with the core.
The strip is lineage zero from the forward arm; these are not consecutive observations
of a spontaneous mutation. Each eligible exact position was tested with all 255 alternative
values against 17 frozen partners. Registered serial assays follow 16 lineages for eight
generations for each forward and reciprocal edit. Sources: `lineage_strips.csv` generated
from `results/capacity/edit_eligibility.csv` and `serial.csv.gz` by `prepare_v62.py`. [manuscript/figures/out/v8/data/lineage_strips.csv; results/capacity/capacity_cases.csv.gz]

**d,** Exploratory sweep of all 256 possible byte values at byte 0 of `ldir32` (L=32,
four-byte core), hexadecimal 00 at the upper left and FF at the lower right. Light-grey
tiles are 157 nonworking values. White teal-outlined
tiles are 88 working values outside
the recorded theorem conditions; they have no payload prediction. The 11
eligible working values have teal fill by predicted d-4, from zero to 28 positions; the
zero-capacity shade is visibly distinct from white excluded interiors. Black
rims mark 04 and 60. Here d is effective displacement modulo the 64-byte ring; eligibility
requires 4<=d<=32, d dividing L, protected-prefix/phase and sufficient coverage conditions.
Working means two-generation normalized culture gain at least 0.3 against fixed panels
of 16 uniform partners. Before and after are the second partner's best-cyclic-shift match
to the original parent, before and after the first offspring reproduces. Gain is
(after-before)/(1-before), averaged over partners with before-match below 0.75. This sweep
does not itself certify partner-universal or serial
inheritance. Source: `dial_coordinates.csv`, prepared by `prepare_v62.py` from
`results/crystal/alleles_ldir32.csv`. [manuscript/figures/out/v8/data/dial_coordinates.csv; manuscript/figures/out/v8/data/methods.csv]

**Figure 4. The environment sorts reproductive designs.**

**a,** Mutation-on L32 competitions between a regenerator R and transmitter T, in benign
or lethal tar. Upper curves are T/(R+T); lower curves are core-bearing cell counts
divided by all 20,000 cells (motif prevalence). Each line is a world median at every archived sample. Dark
lines and filled sparse markers are core C1, lighter lines and open markers historical;
line styles and circle/square/triangle markers identify equal starts (10 worlds),
R introduced at 1% into T (five), and T introduced at 1% into R (five), separately for
each chemistry/cohort. All 40 core C1 raw logs and 40 historical logs are available. core C1 raw
logs were downloaded and reconciled with the frozen dispatch spec, completion markers,
machine core/world/WGSL hashes and all endpoint measurements; no remote simulation was
invoked for this rebuild. Sources: `competition_trajectories.csv`, `competition_medians.csv`
from `prepare_v6.py` and `prepare_v62.py`, with `c1_download_audit.json`.
Core-bearing counts include other byte-0 variants of the shared 5e ed b0 motif, while R
and T follow the frozen displacement classifier; R+T need not equal all core-bearing cells.
Worlds with R+T=0 have undefined relative shares, omitted from ratio medians only.
The inline numeric columns show endpoint median T share and absolute core share for
each starting mixture, with core C1 and archive explicitly paired. [manuscript/figures/out/v8/data/competition_trajectories.csv; manuscript/figures/out/v8/data/competition_medians.csv; manuscript/figures/out/v8/data/competition_endpoints.csv]

**b,** World-level distributions at 300,000 steps: relative T share, motif prevalence,
and distinct transmitter tails. Each chemistry/cohort has 20 worlds. Open markers are
historical and filled markers core C1; circles, squares and triangles denote the same three
starting mixtures as in the inline key. Deterministic horizontal collision offsets
carry no quantitative meaning. One core-C1 benign core-loss world has undefined T/(R+T)
and is omitted only from the relative-share view, leaving 19 defined points there; its
absolute abundance and tail count are zero and remain plotted. Tail diversity is nested
within world, not extra replication. Sources: `competition_endpoints.csv`, `endpoint_positions.csv`. [manuscript/figures/out/v8/data/competition_endpoints.csv; manuscript/figures/out/v8/data/endpoint_positions.csv]

**c,** A numeric lethality dial, preserving all three measures and both cohorts. Each
entry gives core C1 | archive as a pair of values. Open-first and
transmitter-majority counts each use all ten worlds at that probability. Probability p
is the chance that executing a zero halts. Open-first means a first detected heritable
top-three tape at at least 0.5% abundance enters partner instructions in at least 8 of
16 traced encounters; confinement requires zero partner entries in all 16. Transmitter
majority means more than half of sampled heritable cells are transmitters. Nondetections
remain in the count denominators. Confinement timing is a Kaplan-Meier median with
nondetections censored at 300,000 steps, expressed in thousands of steps. The core C1 timing
reversal from p=0.3 to p=1 is retained. Historical DZ2/DZ3 predictions failed while their
core C1 reruns passed; the outcomes remain separate. The environment-to-equilibrium mechanism
is incomplete; no equilibrium curve from the rejected mutation-only model is fitted.
Sources: `lethality_summary.csv`, generated by `prepare_v62.py` from historical and
core C1 dial worlds and checked against `results/tier2/T2_DZ.csv`. [manuscript/figures/out/v8/data/lethality_summary.csv; results/tier2/T2_DZ.csv]

**Figure 5. Different machines, different beginnings.**

**a,** First detected heritable-replicator route, as fractions of all independent worlds,
with integer counts inside every nonzero segment. Z80 Stage G includes 80 worlds across
L=16,20,50,64 (20 per length), all open first; 6502 S1 arm A includes 20 L16 worlds, all
confined first; coherent-fetch 8086 S2 arm A includes 20 L16 worlds, with 2 open first,
12 confined first and 6 right-censored at 3,000,000 world updates. Open and confined
refer to partner instruction fetch, not mechanistic causality. S1/S2 openness requires
partner fetch in at least 8/16 full-encounter traces; confinement requires none in all
16. Z80 G's historical route evidence uses absence of control flow plus partner-dependent
copying, and is not silently relabelled as a newly run S2 classifier. The cohorts have
different observation schedules and horizons. In Z80 G the registered G1(a) open-first
criteria were met at L16 and L50; G1(b) was met at both lengths, G1(c) only at L16,
and the G1 kill criterion did not trigger. These are not one pooled registered verdict
for the 80-world bar. S1 A's full first-confined-plus-loop P1 prediction was not supported
(14/20 have the additional loop criterion); P2 and P4 were supported, P3 offset equality
was untested, and the kill rule did not trigger. The panel's 20/20 confinement count
includes loop and nonloop tapes. S2 R0 passed (14 detections, requirement 10); R1 failed
(2 open of 14, requirement 12); the route kill rule triggered (12 confined, threshold 4).
The registered census-motivated 8086 open-beginning claim is withdrawn. R2 is descriptive
only. Sources: `results/stageG/stageG/stage_g_runs.csv`, `results/stageG/FINDINGS.md`,
`results/s6502/s1_worlds.csv`, `results/s6502/S1.md`, `results/s8086/tables/worlds.csv`
and `results/s8086/S2.md`; generated as `route_by_machine.csv`. [manuscript/figures/out/v8/data/route_by_machine.csv; results/s8086/S2.md; results/s6502/v2/arm_comparison.csv]

**b,** Kaplan-Meier cumulative detection (1 minus survival), using the recorded first
heritable-replicator detection times. All three curves use L16, 128 instructions per
encounter and 20 worlds: Z80 Stage G, 6502 S1 A, and coherent-fetch 8086 S2 A. Direct
labels identify cohorts and their detected/all-world denominators. Steps are world
updates (encounter batches), not CPU instructions. The log axis spans 40 to 12 million.
Each curve extends to its actual observation horizon: 300,000, 10,000,000 and 3,000,000
updates respectively. The six coincident 8086 censoring ticks at 3M are one visible
mark directly labelled “6 censored”; no tick is invented for fully detected cohorts. Event updates are
scheduled detections, not exact births. Protocols and sampling differ; this comparison
is descriptive and supplies no registered directional machine timing claim. Generated
`arrival_worlds.csv`, `arrival_curves.csv`, `arrival_censors.csv`; the 8086 survival
coordinates are checked against the frozen `tables/kaplan_meier.csv`. [manuscript/figures/out/v8/data/arrival_worlds.csv; manuscript/figures/out/v8/data/arrival_curves.csv; manuscript/figures/out/v8/data/arrival_censors.csv; results/s8086/tables/kaplan_meier.csv]

**c,** Compact literal-channel matrix. Width means bytes emitted per push in the
load-literal/push-register channel: Z80 and 8086 width 2, 6502 width 1. Exact periodic
pusher availability is a census statement, not a guarantee of establishment or route
frequency. Z80 has an exact pusher; none was found in the registered 6502 census;
8086 has the `B9 51` (`MOV CX,imm16; PUSH CX`) pusher. The observed first route is open
in 80/80 Z80 worlds, confined in 20/20 6502 A worlds, and predominantly confined in
8086 (12/14 detections; 2 remain open). The teal outlined 8086 row makes the factual
surprise explicit: width 2 and a census pusher coexist with confined beginnings.
This does not promote the infinite periodic width-one obstruction to every finite
ring encounter, or identify opcode width as the unique causal variable. Sources:
`results/literal_channel/CHECK.md`, `results/census8086/table_hits.csv`, the prior
Z80/6502 census evidence, and the route tables. Generated `literal_matrix.csv`. [manuscript/figures/out/v8/data/literal_matrix.csv; results/census8086/table_hits.csv]

**d,** All ten 8086 S2 first-copier tapes consisting of four repeats of a `5F A4 x y`
tile, aligned in their recorded physical phase. Each row shows all 16 hexadecimal
bytes and its full world seed, detection update, and mechanism in all 16/16
random-partner encounters. The repeated teal `5F A4` pair decodes as `POP DI; MOVSB`;
grey x/y positions vary across worlds. Colour identifies shared bytes, not a certified
law core or free payload. Brackets identify all four tiles and position labels identify
0-15. All ten tapes are confined; eight have the descriptive memory-fed one-pass
majority and two a confined copy-loop majority. They are ten of 14 detected worlds,
not all first copiers. The panel itself shows observed bytes and frozen descriptive mechanism labels; it does not depict a provenance certificate or follow-up assay. Source: `results/s8086/tables/worlds.csv`,
generated `first_copier_tiles.csv`. Exploratory, post hoc follow-up identifies the four-byte overlap geometry and transmitted branch alleles (Methods); those results do not alter the panel’s original mechanism labels or the registered law verdict. Raw instruction-start addresses used by the descriptive classifier differ from physical ring positions used to explain the copy. [manuscript/figures/out/v8/data/first_copier_tiles.csv; results/s8086/exploratory/first_family.csv; results/s8086/exploratory/family_transmission.csv]

## Methods

### Execution contracts separate instructions from the world

Zilion separates instruction cores, shared encounter interfaces, execution engines and machine manifests. Population pairing, mutation and scientific classifications remain research code. Manifests bind semantic version, compatibility, chip variant, generated shader and host identities to reset, address mapping, budget, traps and observations. Historical Algocell populations and assays retain their actual kernels; fresh corrected-core reruns are not relabelled historical Zilion executions. Architecture and machine-specific contracts are archived separately from experiment results [algocell_exp/machines/; /Users/neo/repos/zilion/docs/ARCHITECTURE.md].

An encounter places focal A and partner B on a ring, resets the processor, enters A, executes its budget and writes both halves back. Architectural wrapping precedes ring reduction; a wrap need not advance to the neighbouring ring byte for arbitrary ring sizes. The historical Z80 resets registers and flags to zero except SP, set to the largest architectural address aliasing the partner’s last byte. That pointer convention does not itself specify the first written byte. Fetch maps include prefixes, operands and displacements, not merely instruction starts. Repeating block moves consume budget per repetition; I/O reads return zero, writes have no external effect, and no external interrupts occur. Benign zero executes normally; lethal tar halts on a fetched zero opcode. The 8080 subset is an ablation within this core, not independent hardware validation [results/emulator/RING_DIFFTEST.md; algocell_exp/shader/; results/review_r2/RUN_TABLE.md].

The NMOS 6502 starts with clear accumulator/index registers, status `20` hexadecimal and a first push aligned to the partner’s last byte. Zero-page and stack-page architectural accesses map into the ring. Faithful BRK vectoring and halt-before-BRK are separate contracts; JAM halts, without external IRQ/NMI. BFF’s published contract halts on unmatched brackets and pointer run-off; its literal, bandwidth, pointer and halting variants remain separate [results/s6502/CORRECTION.md; algocell_exp/machines/; results/bff/runs.csv].

### The 8086 uses a frozen coherent-fetch abstraction

S2 used the Zilion NMOS core frozen at `39a60429a9b16c77cd845f6087062c4e2cc12c53`, compatibility `0.3`, with `prefetch=coherent` and `hidden_latch=zero_ea`; package status was experimental. Consumed instruction bytes come from current memory. There is no prefetch queue or bus-timing model. General registers, segments and entry IP reset to zero except SP = 1; FLAGS is `F002`, with DF and TF clear. At the registered ring sizes, the first word push decrements SP and writes low then high bytes at ring positions P−1 and 0. The earlier exploratory zero-SP inventory is superseded [algocell_exp/machines/worlds_8086.json; algocell_exp/machines/PROVENANCE_8086.md; results/s8086/prelaunch/freeze.json].

Segment translation and architectural wrapping precede ring mapping; each word byte wraps its offset independently. A REP iteration consumes a budget step and retains decoded operation/source segment between continuations; zero count copies nothing. Register LEA/LES/LDS forms trap. Undefined far-register transfers use the declared fresh zero-EA hidden-state abstraction. Interrupt vectors alias ring memory, external interrupts are absent, input supplies all-one bytes, output is discarded, HLT halts and WAIT without a coprocessor does nothing. These policies matter to evolving programs; coherent fetch is the research contract, not a claim to reproduce hardware-prefetch ecology [algocell_exp/machines/worlds_8086.json; results/s8086/prelaunch/freeze.json; /Users/neo/repos/zilion/docs/I8086.md].

### World sampling dates detection, not birth

The shared soup contains 20,000 cells on a 160 × 125 lattice with reflecting cardinal-neighbour boundaries. Each update draws 8,192 ordered pair slots; collision claims determine active encounters, so draws are not executed-pair counts. The nominal encounter budget is 128 steps. At noise exponent 4, mutation makes 512 uniform byte-replacement attempts per update. Packed-word collisions can lose updates, and replacement may leave a byte unchanged. Applied mutation counts are unmeasured. Pairing races also mean seeds do not guarantee identical whole-world replay [results/review_r2/RUN_TABLE.md; results/s8086/prelaunch/registration_basis.json; results/s8086/prelaunch/freeze.json].

S2’s sole population arm comprised 20 L16 worlds, seeds 8086001–8086020, through 3,000,000 updates. L32 population arm B was withdrawn; pointer randomization was omitted because it lacked a controlled census. L32 remains fixed-input census/control context only. Each world retained 585 scheduled records, including initialization, and complete checkpoints at 0, 20,000, 100,000, 1,000,000 and 3,000,000 updates. Sampling intervals widen according to the frozen schedule; they are not adjusted after detection [results/s8086/tables/worlds.csv; results/s8086/tables/download_hashes.csv; results/s8086/prelaunch/freeze.json].

Detection is the first scheduled top-three exact tape at share at least 0.005 that passes culture, choosing greatest share then lexicographic order. Final random-cell sampling does not retroactively change detection. Times retain preceding-sample intervals. Complete nondetections are right-censored at horizon; missing or corrupt records are incomplete, not biological negatives. Kaplan–Meier summaries use worlds as replicates. Cells, roles, alternatives and descendants are nested observations; none enlarges the number of independently evolved worlds [results/s8086/tables/worlds.csv; results/s8086/tables/kaplan_meier.csv; results/s6502/v2/arm_comparison.csv; REVISION_PREREG.md, S2].

### Culture, carriage and inflow measure different properties

Culture scores normalized gain in best-cyclic-shift similarity to the original parent, `(after−before)/(1−before)`, averaged over informative partners with before-match below 0.75. First offspring reproduce against fresh partners; historical “heritable” means generation-two score at least 0.3, not exact inheritance. S2 used 64 uniform partners per generation, including final-cell cultures. NaN is unassessable, not zero. Its similarity and faithful-offspring fields are reported but do not add undeclared detection gates [manuscript/figures/out/v8/data/methods.csv; results/s8086/tables/copier_tapes.csv; results/s8086/prelaunch/registration_basis.json].

Single-site scans ask whether a reproductively competent mutant transmits its changed byte into at least half of qualifying aligned offspring. A site needs at least half its tested identifiable values to pass. Population classes call confined cells with at most two sites regenerators, three or four intermediate, and at least five transmitters. These thresholded classes do not measure exact capacity. “Unexecuted” means absent from instruction-stream fetches, including operands; it does not exclude data use [results/ladder/LADDER.md; results/population/classes_snapshots.csv; results/crystal/cells.csv].

Historical serial assays follow the centred five-byte mutant window, permitting cyclic alignment and correcting carriage against surviving unmutated controls. Windows already present in wild type are unidentifiable and excluded. The panel uses 16 mutant lineages, 256 controls and eight transfers; alternatives are exhaustive for short tapes and sampled for longer ones. Branching X1 instead gives each survivor two partners, caps populations at 32, and uses four populations for each of 32 alternatives/site plus 64 controls. A population carries when at least half its members carry the marker; site scoring requires the registered 0.5 excess over control carriage among survivors. Fate fractions in Fig. 2a use all original starts. The soup-partner arm pools open-phase worlds rather than the focal neighbourhood; pre-existing matches and reacquisition can inflate carriage [results/serial_retention/sr_summary.csv; results/serial_retention/REPORT.md; results/branching/branching_summary.csv; results/branching/BRANCHING.md].

For deterministic offspring Y = F(x,E), fixing parent and contract gives I(E;Y|X=x) = H(Y|X=x). Exact copying implies zero inflow, but zero inflow does not imply preservation of parental variation. Sampled zero entropy is not partner-universal closure. The historical entropy panel has 256 outputs and an eight-bit sample ceiling, with downward undersampling bias. Fetch confinement, original-partner access proxies, output variation and parental transmission therefore remain separate [theorem: manuscript/MAIN_nature.md, Lemma 7; results/biology/individuality/per_replicator.csv].

### Competition endpoints distinguish relative success from absolute abundance

Registered competitions seed regenerator and transmitter designs in equal mixtures or reciprocal rare invasions and retain mutation-on/off and benign/lethal arms. The designs differ in payload background as well as displacement, so environmental sorting does not isolate selection on capacity. Figure 4 shows L32 mutation-on runs through 300,000 updates: ten equal-start worlds and five per invasion direction for each chemistry/cohort. Relative share is T/(R+T); broader core-motif prevalence uses all cells. Undefined ratios when both classes disappear are omitted only from ratio summaries, retaining absolute loss and zero tail diversity. Historical and core-C1 trajectories are summarized separately at world level [results/offset/offset_switch.csv; results/tier2/scored/E/offset_switch.csv; manuscript/figures/out/v8/data/competition_endpoints.csv].

The lethality dial varies the probability that a fetched zero opcode halts, with nondetections censored at its horizon. Its remaining corrected-core timing reversal is retained. Rejected toxin, backup-core and mutation-only mechanisms do not supply a fitted equilibrium explanation, and exploratory intruder assays identify a lead rather than a completed mechanism [results/tier2/T2_DZ.csv; results/toxin/CAUSAL.md; results/n2/BACKUP.md; results/offset/OFFSET_FLOW.md; results/offset/INTRUDER.md].

### Forward overlap gives a conditional payload law

**Theorem.** On a ring of P bytes, perform ordered writes M[s+d+k] ← M[s+k], modulo P, with 0 < d < P and no interfering operation. Before k=d, reads obtain original source bytes; subsequently each source was last written d steps earlier. Induction gives ancestral coordinate s+(k mod d). With d dividing P and coverage at least P−d, the protected seed tiles the ring; with P=2L and d dividing L, the offspring has the required phase. If a fixed constrained footprint C lies in the protected seed, independent free payload is r=d−|C|. This describes a product family of 256^r sequences, or 8r ideal noiseless bits, not ecological neutrality or measured Shannon capacity [theorem: manuscript/MAIN_nature.md, Proposition 8; results/capacity/geometry_traces.csv.gz].

For the Z80 `XX 5e ed b0` specialization, L≤64, d=XX mod 2L, 4≤d≤L and d divides L. The first instruction must pass control to address 1, preserve the needed source/base state, write only outside the protected prefix, and leave HL≡0 and 256D≡0 modulo 2L, with BC zero or at least 126. Under the declared initialization, 126 block moves remain; the immediate `1e NN ed b0` form leaves 127. Architectural address increments must preserve displacement. Core preservation keeps execution in the copy when overwritten instructions are revisited. Free coordinates then survive jointly and serially; reverse `LDDR` requires another derivation [theorem: manuscript/MAIN_nature.md, audited Proposition 8; results/capacity/candidate_eligibility.csv].

Working core alleles lie outside fixed-footprint payload. Visible sequence period is not displacement. Motif/divisibility screening supplies candidates, not every proof hypothesis. The exploratory population comparison retains its whole-tape gate deficit and additional fetched-site scores rather than redefining the theorem around measured sites [results/crystal/cell_exceptions.csv; results/crystal/offset_examples.csv; results/offset/offset_cells.csv].

### Provenance certificates precede retention assays

A sufficient certificate fixes the admissible parent family, reset, partners, budget, descendant boundary and alignment. Original parent/partner coordinates receive symbolic labels; each pure copy inherits its source’s current label in ordered execution. The proof must establish complete offspring provenance, noninterference under jointly arbitrary admitted payload assignments, viable offspring, preserved entry machinery and membership in the same family after reset. This makes serial induction nonvacuous. One trace or matching byte values alone cannot prove the family claim. Arithmetic encodings and value-dependent programs may transmit information outside this sufficient pure-copy criterion [theorem; results/capacity/candidate_eligibility.csv; results/evolved6502/eligibility.csv; results/s8086/prelaunch/ASTRA_PRELAUNCH_REVIEW.md].

Amended S2 L2 requires the actual sampled baseline to be d-periodic at physical entry zero, with d dividing L, full partner coverage and no original-partner input or interference. It permits original roots R={s,…,s+d−1} inside A. For physical offspring position q, predict a(q)=s+((L+q−s) mod d). Independently extract raw core K, retaining the bound c=1+max(K)≤d, and define source core C_R={a(k):k∈K}, free set F=R\C_R and erased set E=A\R. Hold both K and C_R. The exact free count is d−|C_R|, not necessarily d−c. Mutable erasures are E\K; erased setup coordinates E∩K remain held because their values may govern execution even though their original ancestry disappears. No dependency is reduced modulo d to hide a setup requirement. Source phase and physical entry are distinct; compare physical B directly, without favourable rotations. The certified map composes with itself across transfers [theorem and registered amendment: REVISION_PREREG.md, S2 B2; results/s8086/prelaunch/ASTRA_PRELAUNCH_REVIEW.md; results/census8086/LAW_CONTRACT.md].

### Registered C1 measures capacity after inherited displacement settings

Displacement experiment C1 was registered at `213a653`; eligibility was frozen at `4855ac3`. Its 22 archived parents include fixed representatives and deterministically ordered eligible archive rows, not independent worlds. All 101 settings qualified before retention. The default archived Z80 executor, entry-phase core, dividing displacement, ordered copying, coverage and noninterference were checked against zero plus 16 frozen partners. Initial tails need not be periodic: regeneration is the predicted offspring, not extinction [results/capacity/parents.csv; results/capacity/edit_eligibility.csv; results/capacity/partners.csv; REVISION_PREREG.md, C1].

Forward passages followed 16 lineages through eight generations; reciprocal passages restored the original displacement byte in actual descendants and repeated transfer. Complete core, displacement, offspring and ancestry are retained in 25,856 serial rows. Direct assays used actual generation-eight descendants, testing all 255 alternatives at every free and erased coordinate plus 64 joint assignments against all 17 partners. Deterministic duplicate descendants share execution while retaining lineage identities. All 1,067,608 direct cases and all serial observations matched. These are intervention cases, not encounter counts. Joint sampling is finite; the unrestricted joint claim is conditional mathematics. C1 used local computation with no paid compute [results/capacity/serial.csv.gz; results/capacity/capacity_cases.csv.gz; results/capacity/CAPACITY.md; results/capacity/bursts.csv].

### Registered C2 uses the complete evolved 6502 panel

C2 shares registration `213a653`. The original extractor and 714-role ledger were frozen at `943aeee`; unchanged extraction extended to 942 roles at `657eb7e` before interventions on newly recovered top-three roles. The descriptive-copier inclusion clarification at `f57ad24` preceded new assays. Corrected S1 frozen32 primary calls determine the descriptive final-copier panel; every registered sampling role remains in the geometry ledger. Sensitivity evaluations do not create cells. All raw logs matched their frozen input hashes, with no substitute tapes [results/evolved6502/coverage.csv; results/evolved6502/inputs.csv; results/evolved6502/RETENTION.md].

The complete panel contains 62 first detections, 640 final random draws and 240 final top-three roles from 80 worlds. It has 861 distinct tapes and 42 eligible roles. Footprint c counts constrained coordinates, not necessarily a contiguous prefix. Ordered provenance admits pure-copy channels and rejects or leaves unresolved unknown arithmetic, reverse/nonconstant motion, partner inflow, interference and uncertified control. Using each world’s frozen 6502 machine at budget 128, every admitted role received all 255 alternatives at each free/erased coordinate and 64 joint assignments against zero plus 16 uniform partners, followed through eight fresh-partner generations. Complete physical offspring and source labels are scored without similarity gates or cyclic realignment. There were 91,428 intervention cases and 731,424 serial observations, without mismatch. A separate descriptive serial scan uses marker thresholds on ineligible copiers: two first and two final roles per arm, 32 alternatives/site, four mutant lineages and 64 controls. It neither tests the exact law nor supplies a registered directional prediction. C2 used no paid compute [results/evolved6502/panel.csv; results/evolved6502/eligibility.csv; results/evolved6502/geometry.csv; results/evolved6502/law_verdict.csv; results/evolved6502/descriptive_panel.csv; results/evolved6502/exact_cases.csv.gz; results/evolved6502/exact_serial.csv.gz; results/evolved6502/bursts.csv].

### Literal channels constrain availability within a stated grammar

**Theorem.** A repeated pair with distinct one-byte load/push opcodes loads w literal bytes into one register and pushes a fixed permutation without changing it, using a separate stack pointer that moves by −w per push. Under contiguous noninterfering writes and the prescribed executed stream, input/output periods are w+2 and w. Fine–Wilf requires N≥2w+2−gcd(2,w) for finite-window equality. Odd widths then force identical opcodes, a contradiction; even widths permit compatible alternating constructions. Descending stacks reverse block order in ascending memory, not internal byte order. Placement, bootstrap and coverage remain separate [theorem: results/literal_channel/CHECK.md; results/literal_channel/pair_layouts.csv; results/literal_channel/finite_bound_counterexamples.json].

**Theorem.** For width-one periodic mixtures through a shared register, a minimal decoder cycle visits at most d residues of primitive input period d. If it contains loads and pushes, emitted length n_P satisfies 0<n_P<t≤d, incompatible with d dividing n_P for an equal infinite stream. Multiple opcode identities do not change this. The degenerate all-push writer initialized with an opcode-valued register is excluded. Width-three mixtures provide counterexamples to extending parity beyond repeated pairs. A finite width-one factor witness exists but fails archived zero-partner bootstrap. Only three of the five nontrivial Z80 period-two writers are literal; the others are memory-fed. None of these statements proves an evolutionary arrival order or excludes every finite 6502 copier [theorem/construction: results/literal_channel/verification.csv; results/literal_channel/finite_search.csv; results/literal_channel/6502_finite_harness.json; results/literal_channel/census_audit.csv].

### The causal census changed descriptive labels, not outputs

The validated 8086 census evaluated all 256 period-one and 65,536 period-two words at L16/L32 and budgets 128/512 against zero plus 16 frozen partners, retaining constants and reversed phases, with no unevaluated words. Use the final causal correction, not earlier labels in the census narrative or partial zero-SP inventory. Its audit covers 1,246 settings and 21,182 encounters; 1,247 labels changed without changing outputs or matches [results/census8086/table_main.csv; results/census8086/table_hits.csv; results/census8086/descriptive_correction_counts.csv; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md].

The classifier examines execution through the last partner write. Every claimed store, including neutral read-modify-write, must pass isolated `55` and `AA` source interventions with matching ordered addresses through that write and the expected source-dependent value. Later divergence is allowed. Literal provenance, paired memory copying and confined copy loops are separate; loops require repeated full instruction-start addresses, not merely mapped ring positions. Mixed, unsupported, failed, vector-dependent or overflowing cases remain “other.” Trace memory and original records must agree with uninstrumented execution. Full-encounter partner fetch and pre-write access proxies are separate from this causal classification [results/census8086/MECHANISM_CONTRACT.md; results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md; results/s8086/tables/judged_encounters.csv.gz].

Repeated classifier corrections undermined the rationale for R2’s numerical predictions, so R2 occurrence and persistence were withdrawn prospectively before population data; they remain descriptive. R0/R1 use independent culture/fetch endpoints. The `07 A5` witness remains causally memory-fed in all 68 audited contexts and open in all 64 random-partner contexts. At budget 128, `B9 51` bootstraps against zero and copies exactly into 13/16 random partners at L16 and 6/16 at L32, so it is partner-dependent; `FF 30` and `AD 50` are not exact zero-partner bootstrappers under this reset. Available mechanisms are not establishment probabilities [results/s8086/tables/census_07a5.csv; results/census8086/table_hits.csv; REVISION_PREREG.md, prospective R2 withdrawal].

### Registered S2 keeps route and law verdicts separate

R0 requires at least 10 detections among 20 worlds. Given that sufficiency, R1 requires at least ceil(0.8D) open first copiers; the kill rule triggers at ceil(0.25D) confined first copiers. Open means partner fetch in at least 8/16 complete traces, confined none in all 16, and intermediate histories remain in D. Observed D=14, open=2 and confined=12 give R0 pass, R1 fail and the route kill. R2 records occurrence in 1/20 worlds and persistence in 0/20, without confirmatory cutoffs. Confinement timing has median 975,000 updates, with eight censored worlds; it is not a causal-closure or loop endpoint [results/s8086/tables/worlds.csv; results/s8086/tables/persistence.csv; results/s8086/tables/kaplan_meier.csv; results/s8086/S2.md].

Law candidates comprise first detection, first confinement, final top-three tapes and 64 uniformly sampled final cells per world, preserving cell multiplicity. Eligibility was committed at `8971d0` before evolved interventions. L1 admits only the verified six-byte MOV-DI/DEC-CX/REP-MOVSB-or-MOVSW family; L2 uses the amended certificate above, with separate with/without-L1 reporting and per-family caps. Of 1,337 distinct genotype-world candidates, 779 were culture-positive. L1 excluded all; L2 excluded 55 for original-partner reads and left 724 unresolved, including 708 at POP ES. No cap excluded a case, and no evolved law intervention occurred. L1 is untested; L2 is untested with support suspended. The frozen reporter’s `incomplete/invalid` extraction label does not invalidate the completed route cohort [results/s8086/tables/law_candidates.csv; results/s8086/tables/law_denominators.csv; results/s8086/tables/law_worlds.csv; results/s8086/S2.md].

L1 requires the sampled baseline to be d-periodic at physical entry zero, 6≤d≤L with d dividing L and even d for MOVSW, reset source/segments, sufficient ordered forward coverage, preserved core and no interfering write or original-partner read. Each family could select at most 16 genotype-world cases per world, taking eligible first/closed/final roles before keyed random-cell candidates. The registered design tests all 256 values at free and mutable erased coordinates, 256 counterbalanced plus 256 seeded joint assignments, all 17 census partners, and eight serial generations for singles/joints. A reproducible mismatch remaining independently eligible fails the application; unresolved certificates suspend support. These planned evolved assay denominators remained empty [results/census8086/LAW_CONTRACT.md; results/s8086/tables/law_outcomes.json; REVISION_PREREG.md, S2].

Constructed controls are separate evidence. The preceding census qualified 340 geometry/partner/budget rows; its six-byte construction gives 2/10 free coordinates at L16 and 2/10/26 at L32 under the admissible displacements [construction: results/census8086/table_law.csv]. All 10 byte/word REP controls at the registered budget completed with zero mismatches in 1,185,792 encounters, including fixed-input L32 geometries. They test complete physical offspring, ancestry and serial reset closure; they supply no evolved support [construction: results/s8086/tables/constructed_controls.csv; results/s8086/tables/control_hash_verification.csv].

### Operational amendments preserved the scientific contract

A failed detached-app stop was repaired by making the command noninteractive. World 8086001 subsequently completed computation but hard-link publication failed; before reading biological output, the attempt was designated noncanonical and retained by opaque hashes. Publication changed to same-directory atomic rename, and the same seed restarted from initialization. Checkpoints lacked full continuation state, so no trajectory was spliced. A separate amendment authorized parallel dispatch and the measured 9,000-second world body limit. After all worlds completed, transient ledger lock contention falsely terminated scoring; bounded retries and diagnostic cause labels repaired the control path, and only scoring onward resumed. These changes preserved scientific source, machine, sampling, eligibility and endpoint definitions. Failed attempts and charges remain archived; nondeterministic restart does not promise the same trajectory [results/s8086/prelaunch/S2_OPERATIONAL_AMENDMENT_2026-10-12.md; results/s8086/prelaunch/S2_VOLUME_PARALLEL_AMENDMENT_2026-10-12.md; results/s8086/prelaunch/S2_SCORING_ONWARD_LOCK_AMENDMENT_2026-10-12.md; results/s8086/provenance/run_provenance.json].

The final reconciliation recorded $13.22378749 in app-attributed charges, reported as $13.22, including preparation and earlier attempts. Billing may lag; this is a metered reading, not a settled invoice. The conservative retained bound was $77.32835455 against the amended $95 cap. All attributed apps were provider-confirmed stopped. Source, world/checkpoint identities and control hashes were verified; unchanged reporters generated the final verdicts [results/s8086/provenance/final_spend.json; results/s8086/provenance/verification.json; results/s8086/tables/control_hash_verification.csv].

### The 8086 follow-ups are exploratory and post hoc

Post hoc scans covered both pusher phases in all scheduled top-three records and complete checkpoints. Encounter assays planted tiles in focal and receiving roles against sampled soup and matched uniform partners. Absence of observed seeds and receiving-role vulnerability motivate an arrival/establishment interpretation, without identifying historical cause, unseen births or population growth. Founder-byte transmission varied homologous positions across the repeated tile; law erasure tests instead altered single physical coordinates [exploratory: results/s8086/exploratory/motif_scan.csv.gz; results/s8086/exploratory/encounter_fates.csv; results/s8086/exploratory/family_transmission.csv].

A separate extended dependency verifier re-extracted all 779 culture-positive candidates, admitting 600 and leaving 72 excluded and 107 unresolved. The purposive 32-genotype panel represented 20 worlds; predictions were committed at `5bbc9d7` before assays, but extraction and selection remain post hoc. The frozen assay was reused with the extended verifier and 64 independent joint assignments in place of 512. All-value single/mutable-erasure substitutions, 17 census partners and eight serial transfers yielded 2,224,640 encounters without mismatch. For d=4, free payload is empty and joint assignments repeat baseline; for d=16, erased coordinates are empty. CPU audits verified hashes, source maps and denominators. No new world or paid remote computation was run, and the registered verifier and verdicts were unchanged [exploratory: results/s8086/exploratory/eligibility_counts.csv; results/s8086/exploratory/law_panel_predictions.csv; results/s8086/exploratory/law_assay_denominators.csv; results/s8086/exploratory/verification.csv; results/s8086/EXPLORATORY.md].

### Validation and chronology retain failures and amendments

Validation distinguishes instruction, ring-encounter, isolated-reproduction and ecological levels. Z80/6502 vectors are emulator-generated; 8088 vectors are hardware-recorded. The latter give 3,126,917 raw passes among 3,137,000 cases, with 10,083 expected hidden-latch divergences and none unexpected; queue state and timing are outside scope. A same-core CPU implementation is not an independent oracle, and deterministic encounter replay does not establish deterministic soup replay [manuscript/figures/out/v8/data/validation_fractions.csv; results/emulator/RING_DIFFTEST.md; algocell_exp/machines/PROVENANCE_8086.md].

I1 compared historical compatibility C0 with corrected classic core C1 and vendor-specific core C2, distinct from experiments C1/C2. It flagged 47/690 and 114/690 conditions respectively, with 407/417 isolated combinations identical in each comparison; the all-unaffected criterion failed. Omitted named Stage-E cases were completed locally with the sealed engine. Stochastic assays use case-index streams, not archived encounter streams. I1-T2 retained 19/22 claim verdicts but changed four of 36 components, including historical DZ2/DZ3 failures becoming passes and Stage-L L2 becoming not applicable at L16 and not met at L20, while its joint kill still fires. Preserved verdicts include negative outcomes. The Stage-L dispatcher’s missing-snapshot alerts were retained; frozen scorers used complete required records. Fresh nondeterministic worlds do not isolate a causal core-repair effect [results/impact/decisions.csv; results/impact/assay_identity.csv; results/impact/supplement_manifest.json; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv; results/tier2/stageL_diagnosis.json].

The SI chronology keeps discovery definitions separate from confirmation. X1’s carriage prediction failed; the first X3 archive existed before registration, whereas its replication was prospective; N6’s failed component remains failed. The validated 6502 census preceded S1, while the literal theorem is retrospective. S1 correction retains unsupported P1, untested P3 and the registered-versus-executed partner-count/source discrepancies. C2’s later exact-law pass does not rescore S1. Crystal, ladder and the 8086 follow-ups are exploratory. S2’s final scientific amendments preceded population data; operational recovery records identify subsequent changes without rewriting prior history [REVISION_PREREG.md; results/lod_n6/N6_SCORE.md; results/s6502/CORRECTION.md; results/evolved6502/coverage.csv; results/s8086/prelaunch/ASTRA_RECHECK3.md; results/s8086/exploratory/publication_archive.csv].

## Extended Data and SI map

This map assigns the supporting material for review. It does not assert that every Extended Data display has already been rendered. The main figures are the existing v8 assets; completed results below are not prospective placeholders.

| Display | Content, status and source family |
|---|---|
| Extended Data 1 | Validation by instruction, encounter, assay and ecology; expected divergences, mutation/collision disclosure, I1 and I1-T2 verdict ledgers, omitted-case supplement and Stage-L dispatcher diagnosis. Historical and corrected-core outcomes remain separate. `results/emulator/`, `results/impact/`, `results/tier2/`, `algocell_exp/machines/`. |
| Extended Data 2 | Complete historical serial/registered branching maps; extinction versus surviving erasure, rare core alleles, identifiable-window denominators, control carriage, partner panels and ordered erasure ancestry. `results/serial_retention/`, `results/branching/`, `results/pe/`, `manuscript/figures/out/v8/data/erasure_ordered.csv`. |
| Extended Data 3 | Registered displacement experiment C1: frozen eligibility, settings, reciprocal passages, actual descendant haplotypes, causal capacity and joint/erasure controls. Exploratory dial and sampled population-score exceptions appear separately. `results/capacity/`, `results/crystal/`, `results/offset/`. |
| Extended Data 4 | Reset dependence, random-cell versus modal sampling, class denominators, marker failures and exploratory detection order. Sampling order does not establish ancestry. `results/r4/`, `results/population/`, `results/marker_check/`, `results/ladder/`. |
| Extended Data 5 | Full environmental competitions: historical/core-C1 trajectories, relative and absolute abundance, undefined ratios, core loss and tail diversity. Registered rejected mechanisms and exploratory intruder interactions remain distinct. `results/offset/`, `results/tier2/scored/E/`, `results/toxin/`, `results/n2/`. |
| Extended Data 6 | Validated finite 6502/8086 censuses and contract sensitivities; corrected literal-channel proofs and finite counterexamples; constructed controls; S1 correction and complete registered C2 eligibility/retention panel. Earlier 8086 mechanism labels are superseded by the causal correction. Pointer randomization is unmeasured and was not an S2 arm. `results/census6502/`, `results/census8086/`, `results/literal_channel/`, `results/s6502/`, `results/evolved6502/`. |
| Extended Data 7 | BFF literal, bandwidth, pointer and halting controls, including open-fetch/zero-inflow dissociations and failed registered components. `results/bff/`, `results/bff_dials/`, `results/biology/individuality/`. |
| Extended Data 8 | Genealogy synthesis: discovery definitions versus N6 confirmation; first founders versus later conversions; functional donor definitions, correction history and the failed N6 component. `results/lod/`, `results/lod_n6/`. |
| Extended Data 9 | Registered S2 world routes, censoring, descriptive R2 profiles, candidate/exclusion ledger and unresolved L1/L2 certificates, followed by explicitly separate exploratory/post hoc motif scans, encounter fates, family traces, admissions and exact-law panel. Operational attempts and accounting accompany reproducibility, not biological replication. `results/s8086/tables/`, `prelaunch/`, `provenance/`, `exploratory/`. |

Supplementary Methods should carry the full overlap and literal-channel proofs; reset, trap, fetch and budget contracts; provenance transfer rules and eligibility/exclusion dictionaries; assay random streams and source-map conventions; registrations and dated amendments; chronology and complete verdict tables; genealogy corrections; the early length, ablation and padding atlas; and technical detection controls. The C1/C2 certificates and S2’s empty registered eligible tables are archived evidence, not pending analyses. No general machine-checked proof for arbitrary programs is claimed.

Supplementary tables should preserve independent-world counts separately from sampled cells, genotype-world candidates, sampling roles, alternatives, partner challenges and generations. The causal census correction, historical failures, C1/C2 registration and eligibility freezes, S2’s pre-population R2 withdrawal, source-root amendment, operational recovery and post hoc prediction commit belong in the chronology together. Figure-source tables and hashes provide the link back to every reported number. Withdrawn interpretations remain corrections; the map supplies no universal succession, general parity, neutral-manifold, selection-for-evolvability or open-endedness claim.

## Data and code availability

Review materials are archived in this repository under `experiments/results/` and `experiments/manuscript/figures/out/v8/`. Each result family retains generated tables, input identities, code/protocol references and status. The manuscript’s source brackets identify the relevant records. S2 additionally retains exact frozen reporter outputs, source and download inventories, failed-attempt provenance, eligibility release, operational amendments and billing reconciliation. The exploratory publication archive preserves historical launch records without rewriting them [results/s8086/provenance/artifact_manifest.json; results/s8086/provenance/verification.json; results/s8086/exploratory/publication_archive.csv].

Capacity and retention reports are generated by `capacity_assays.py` and `evolved6502_retention.py`. S2’s registered presentation is regenerated by `results/s8086/provenance/generate_report.py`; post hoc analysis and its independent hash audit are in `exploratory8086.py`, `exploratory8086_law.py` and `exploratory8086_verify.py`. Report generation and new execution are different operations. The archive includes the pre-assay prediction identities needed to establish their ordering. Some full raw logs/snapshots are local or externally stored rather than committed; their input paths and expected hashes are recorded in the corresponding manifests. Local absolute Zilion paths are review-source locations, not public accessions.

The main figure inputs, plotting tables, legends and technical manifest are in `manuscript/figures/out/v8/`. Reproduction commands are documented in `FIGURES_V8.md`, using `prepare_v8.py`, `build_v8.py`, `legends_v8.py` and `verify_v8.py`. The manuscript legend text harmonizes the frozen v8 displays with the later exploratory report; no exploratory 8086 law points have been added to Figure 3 [manuscript/figures/out/v8/FIGURES_V8.md; manuscript/figures/out/v8/qa_manifest.json].

Public deposition and a release-wide manifest catalogue remain pending, as listed at the top, together with the standalone encounter benchmark package/table. Submission should replace those release placeholders with durable accessions and include the raw records or durable retrieval locations to which their hashes refer. No missing accession is presented as an existing public release.
