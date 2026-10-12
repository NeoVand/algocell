# Title candidates (decided last)

- Reproduction can erase its own variation
- An inherited copying rule controls what survives reproduction
- The copying mechanism sets the capacity for heredity

# Working title: What spontaneous replicators pass on

*Manuscript v6, 2026-10-12. Abstract and main text: Claude's narrative pass over Astra's v5, keeping v5's numbers
and sources. Figure legends, Methods and the ED/SI map: Astra's v5, unchanged. Source brackets follow
`manuscript/NUMBERS_INDEX.md`. `[fill: …]` marks a missing table, and PENDING marks work without an outcome. Every
sentence is one of: theorem, registered result, exploratory analysis, construction or pending. Methods gives the
status of each.*

## Abstract

Darwinian evolution needs more than reproduction: descendants must inherit the differences on which selection acts.
How a copying process first comes to preserve differences, rather than merely making more of itself, is a central
question for the origin of life. However, in systems where replication arises spontaneously, it has not been possible
to measure what reproduction passes on, or to identify the mechanism that decides. Here we introduce Zilion, a
validated multi-processor execution substrate, and use it to intervene on replicators that evolve in soups of random
machine code. Lineages of the first Z80 replicator survive while losing all 512 tested single mutations. We show that a copy's
overlap with itself determines which parental bytes reach descendants, and that the displacement setting this overlap
is itself inherited, so that one copying rule controls how much else is inherited. Changing the chemistry of the
environment shifts which designs prevail. Closure to the environment and heredity of variation are thus distinct
thresholds, giving Woese's Darwinian threshold an experimental form.

## Main text

### Introduction

Natural selection acts on differences, so a lineage can evolve only if reproduction preserves the differences that
arise within it. This requirement separates replication from heredity. A crystal, an autocatalytic network or a
self-reproducing automaton can make more of itself without carrying forward the particular variants that appear in its
parents [von Neumann 1966; Szathmáry & Maynard Smith 1997]. Compositional replicators that reconstruct a preferred
state have limited heredity for exactly this reason [Vasas, Szathmáry & Santos 2010; Vasas et al. 2012]. Woese placed
the start of Darwinian evolution proper at a threshold, the point where cells became integrated enough that vertical
descent dominated the horizontal exchange of an earlier communal regime [Woese 2002].

Artificial chemistries of interacting programs have made the origin of replication itself observable. In soups of
random programs that execute on each other's memory, self-replicators arise without any fitness function, and
replicating architectures succeed one another as the soup evolves [Agüera y Arcas et al. 2024; Cicala et al. 2026].
These systems show how replication begins. They have not been used to ask what the resulting replicators pass on.

That question cannot be answered by watching. The most abundant genotype in a soup can be a copier that restores its
own pattern so effectively that it erases every variant that arises in it. Faithful copying of an unmodified parent
says nothing about copying of its mutants, and the reappearance of a sequence may reflect transmission,
reconstruction or recapture from the surroundings. Distinguishing these requires intervention: introduce a difference
into a parent, let reproduction proceed, and follow the difference separately from the lineage. It also requires a
substrate in which every copying operation can be traced and the same experiment repeated on different processors.

Here we build that substrate and use it to show that heredity in spontaneous replicators is a measurable property of
the copying mechanism. Zilion executes real 8-bit and 16-bit instruction sets (the Zilog Z80, the MOS 6502 and the
Intel 8086) under a versioned semantic contract validated against hardware-recorded instruction tests. In Z80 soups,
we find that:
- the first replicators survive while erasing every tested mutation;
- a single geometric property of the evolved copy explains which parental bytes reach descendants;
- the parameter that sets this geometry is itself inherited;
- the environment determines which hereditary designs prevail.

Comparing processors separates how replication begins from what it transmits.

### Reproduction as an experiment

Zilion separates a processor's instruction semantics from the world in which programs meet (Fig. 1a). An encounter
places two tapes of L bytes on a ring of 2L bytes, resets the processor to a declared state, executes a bounded number
of instructions starting in the first tape, and writes both tapes back. The soup supplies encounters between lattice
neighbours and background mutation, and nothing else: there is no fitness function and no replacement of unsuccessful
programs. A program reproduces only if its execution writes, into some partner, a tape that reproduces in turn.

The contract specifies everything a reproductive claim depends on:
- address reduction on the ring and the initial registers and stack phase;
- what counts as one unit of execution;
- the handling of undefined instructions and interrupts.

Cell boundaries, execution time and reset state are supplied by the world. The code that exploits them evolves. This
distinction matters because an evolved copier can depend on a register value supplied at reset as strongly as on an
instruction encoded in its own tape (Methods). We validated each processor core against hardware-recorded single-step
instruction tests. Re-running the registered experiments with a corrected Z80 core changed 4 of 36 registered
components, none of which reverses a conclusion reported here; we report the corrected values
[results/impact/README.md; results/tier2/T2.md] (Methods).

Four measurements, kept separate throughout, turn a soup into an experiment (Fig. 1b,c):
- **Culture test:** does a tape's offspring itself reproduce?
- **Trace:** does execution stay within the replicator's own bytes?
- **Inflow:** how much does the offspring depend on which partner was overwritten, with the parent held fixed?
- **Transfer assays:** does a deliberately introduced parental difference persist through repeated reproduction?

These properties come apart in both directions. A program can read a partner's bytes without executing them, and can
execute partner addresses after overwriting their contents. So none of them stands in for another (Methods).

### Lineages survive while alleles vanish

In all 80 worlds of the registered benign-chemistry cohort, the first replicator is open: its execution runs on into
the partner it is overwriting [results/inflow_insoup/INFLOW_INSOUP.md, X2-2]. Its most common form tiles the tape with a two-byte word such as
`01 c5` [results/stageG/stageG/NUMBERS_G.md], which is both the instruction `LD BC,$01C5` and the literal data that the following `PUSH BC` writes into
the partner's memory, as first described by Agüera y Arcas and colleagues [Agüera y Arcas et al. 2024]. Every byte of
this replicator is, in turn, an instruction and the datum it copies.

To ask what such a replicator transmits, we performed a registered branching-transfer experiment (X1), designed so
that a mutation could not be lost merely because its lineage died out (Fig. 2a). Each surviving member of a
population reproduced against two fresh partners per generation. Under uniform random partners:
- **Unmutated lineages:** 48 of 64 pusher populations survived for eight generations.
- **Mutant lineages:** none of 512 single-site mutants left a population still carrying its mutation at generation
  eight. Mutant populations did not simply die out: 38% of them were still alive at generation eight, none carrying their
mutation.

The registered prediction that the open replicator would retain heritable variation (X1-2) therefore failed, and we
report it as failed [results/branching/branching_summary.csv, pusher16, arm U; results/branching/SUPP_TABLE.md;
REVISION_PREREG.md, X1]. The replicator persists because it rebuilds its form, not because it passes on what it
carries.

Two closed replicators from the same soups give contrasting results under the identical protocol (Fig. 2a,b).
- **Two regenerators** (one returning control to itself, one using block copying) preserved every unmutated
  population and scored no transmissible site.
- **An evolved block-copy transmitter** carried nine scored sites through all eight generations
  [results/branching/branching_summary.csv, ret16, ldir32, genome20_s6003].

A scored site is a demanding criterion: most alternative values at that position must persist. Rare variants can
still slip through; the 64-byte pusher retained 5 of 2,048 tested mutant types with uniform partners
[results/branching/SUPP_TABLE.md]. The contrast is therefore between copiers that preserve broad classes of
alternatives at specific positions and copiers that preserve almost none, not between all and nothing.

These replicators differ in what they preserve, and none differs much in how well it reproduces. That separates a
property that abundance and copying fidelity cannot reveal. It also raises the question of what, in the copying
operation, decides which differences survive.

### Overlap sets what is inherited

The answer for the most tractable evolved copier is geometric (Fig. 3a). Consider a copy that moves bytes around the
ring, each byte written a fixed displacement d ahead of where it was read.
- **The first d reads** obtain original parental bytes.
- **After that, the read pointer reaches bytes already written,** and from then on it copies copies. The first d bytes
  are therefore propagated around the entire ring, and every byte beyond them is overwritten before it can be read.
- **A small displacement** regenerates a short pattern regardless of what else the parent held. **A displacement that
  spans the whole parent** reproduces it entire.

If the copying machinery occupies c bytes inside the first d, the number of parental positions that can vary freely
and still be transmitted is r = d − c (Methods). The proof follows the provenance of every byte of the final ring back
to its source and holds for jointly arbitrary payload values. Its hypotheses are stated in Methods:
- the direction of the copy;
- coverage;
- that d divides the tape length;
- protection of the copying core.

The Z80 soups evolve exactly this machine. In the four-byte core `XX 5e ed b0`, the first byte executes as a harmless
instruction and is then read as data by `LD E,(HL)` to set the displacement; `LDIR` performs the overlapping copy.
The variant `1e NN ed b0` carries the displacement as an immediate operand. Cicala and colleagues identified this
compact core and noted its partner offset [Cicala et al. 2026]; what the theorem adds is its hereditary consequence:
r = d − 4. The core occurs in 2,347 of 15,655 sampled confined heritable cells, a recurring architecture rather than
the whole ecology [results/offset/CORE_FAMILIES.md].

Where the theorem's conditions apply, it predicts the measurements almost without exception.
- **Unexecuted positions:** among 2,343 evolved cells whose displacement divides the tape length, 2,342 carried
  exactly d − 4 scored unexecuted transmissible positions.
- **Total scored sites:** in all 53 cells whose total exceeded d − 4, the excess lay at executed positions that
  tolerate some alternative values.
- **The one deficit:** the single exception preserves its predicted payload in targeted transfers but fails the
  whole-tape similarity gate of the population assay [exploratory: results/crystal/cells.csv,
  results/crystal/cell_exceptions.csv, results/crystal/offset_examples.csv; results/offset/offset_cells.csv].

The law also corrects a tempting intuition: what matters is the displacement of the copy, not the visible period of
the tape. One evolved tape repeats every four bytes, all of them core, yet carries four transmissible positions,
because its copy displacement is eight [exploratory: results/crystal/cell_exceptions.csv, K, L = 32, seed 5006].
Heredity belongs to the family of sequences a copying operation preserves, not to the appearance of any one member.

The same operation explains the threshold that the first replicators failed to cross. Under the theorem's hypotheses,
every byte of the offspring comes from the parent's protected prefix, whatever the partner held. A regenerator is
therefore closed to its environment, contributing no partner information to its offspring, while erasing nearly all
parental variation. Closure removes an external source of difference. Heredity preserves an internal one.

### The copying rule is inherited

If the displacement is encoded in the replicator's own bytes, it should be copied with the core, and a change in it
should be inherited. Archived serial-transfer interventions on an evolved regenerator, ldir32, show this directly
(Fig. 3c). Changing its displacement byte from `04` to `08` or to `60` produced lineages that still carried the new
value after eight transfers [results/serial_retention/sr_mutants.csv.gz, ldir32, gen 8, pos 0, val 8 and 96]. Under
the theorem these changes raise the displacement from 4 to 8 and 32 bytes, and the predicted free payload from zero to
4 and 28 positions. A copier that inherited nothing can thus, by one inherited change to its copying rule, become one
that inherits most of its tape.

An exhaustive exploratory scan of the displacement byte shows how wide this dial is [exploratory:
results/crystal/alleles_ldir32.csv].
- **Working values:** 99 of the 256 values give a working replicator.
- **Inherited values:** 50 of those are carried into aligned offspring.
- **Range:** among those meeting the theorem's eligibility checks, the displacement ranges from 4 to 32 bytes, so the
  predicted payload ranges from none to 28 positions.

The dial is not universal. An evolved transmitter that copies backwards with `LDDR` lies outside the forward theorem
and has only 15 working values [results/crystal/alleles_genome20_s6003.csv]. Measuring the descendants' expanded
capacity directly, with reciprocal edits, is the next step [PENDING; fill: targeted descendant-capacity table].

This makes concrete an old idea in an evolved form. Von Neumann's self-reproducing automaton needed a description
that is both copied and used [von Neumann 1966]. The displacement byte is used to construct the copy and is copied
along with it, and in being inherited it sets how much of the remaining tape is inherited too. It is a heritable trait
whose phenotype is the capacity for heredity itself. We do not claim that selection favours it for that reason:
whether a larger payload benefits a lineage depends on the environment, to which we now turn.

### The environment sorts hereditary designs

Without mutation, mixtures of regenerators and transmitters drift, with no consistent winner
[results/tier2/scored/E/REPORT.md, mutation off]. With mutation, the soup does not treat them equally. In
registered competitions at tape length 32, with mutation on and equal starting shares of the two designs, the
chemistry of the soup's abundant by-product decided the outcome (Fig. 4a,b) [results/tier2/scored/E/REPORT.md;
results/tier2/T2_COMPONENTS.csv, E-L32L].
- **Benign zeros:** where the zero byte executes harmlessly, transmitters ended at a median share of 0.05 among
  core-bearing cells.
- **Lethal zeros:** where fetching a zero halts execution, they ended at 0.76, and all ten worlds finished above
  one half.

Starting from either design invading the other, populations converged on the same environment-specific share:
0.09 and 0.08 under benign tar, and 0.81 and 0.77 under lethal tar [results/tier2/scored/E/REPORT.md, L = 32]. That
is the signature of an equilibrium set by the environment, not of a race won by the first arrival.

A graded dial of zero-lethality reproduced the shift: transmitter-majority populations formed in 8 of 10 fully
lethal worlds [results/tier2/T2_DZ.csv]. The same dial ends the open beginning. Under lethal tar the first detected
replicators were already closed in all ten worlds, and no open phase was observed [results/crystal/first_cells.csv, Z80
I] (Fig. 4c).

The environment therefore sets the route through the two thresholds:
- **Benign worlds** are typically entered by an open copier, close and then remain dominated by regenerators. In an
  exploratory reanalysis, regenerators were detected before transmitters in 30 of the 42 benign worlds in which both
  appeared [exploratory: results/ladder/LADDER.md].
- **Lethal worlds** begin closed, with first tapes that still carry little payload, and are later taken over by
  designs that carry much more [results/crystal/first_cells.csv].

Why lethal tar favours transmitters remains open.
- **Rejected mechanisms:** registered tests rejected a toxin mechanism, backup cores and isolated mutational flows
  [REVISION_PREREG.md, E-mech; results/offset/OFFSET_ENCOUNTER.md].
- **Where the answer probably lies:** in encounters with core-free intruders, where the two designs fare differently.

Because a tape is both a copier and the material that other programs execute and overwrite, what a replicator
preserves changes what its neighbours encounter. A mechanism for the balance will have to predict how it moves after
an intervention.

### Other machines, other beginnings

A second processor tests which of these findings belong to the Z80. The 6502's only literal channel loads and pushes a
single byte, and we prove that no periodic stream of one-byte literal loads and pushes through one register can write
itself (Methods). This theorem was derived after the 6502 experiment and explains it in retrospect.
- **Before any soup:** an exhaustive census registered that no 6502 tape of period one to four writes itself
  unaided [results/census6502/table_main.csv].
- **In the registered soups:** every one of 20 first replicators in the primary arm was confined from the start.
  Replication arrived after a median of 1,125,000 steps, against hundreds of steps on the Z80, although the two
  machines differ in more than their literal channels [results/ladder/LADDER.md, closure cohorts]. Only 14 of the 20 met
  the registered loop criterion, which missed forward branches that wrap around the ring, so that component of the
  prediction (P1) failed as registered [results/s6502/v2/S1_v2.md; results/s6502/CORRECTION.md].
- **Copying law:** constructed 6502 copy loops reproduced the predicted d − 6 payload in all 296 qualifying cases
  [results/census6502/table_closers_law.csv]. Whether evolved 6502 copiers obey it remains to be measured
  [PENDING; fill: evolved 6502 retention panel].

The route into replication therefore depends on the instruction set, while the copying law, where its conditions
hold, does not.

The 8086 provides the prospective test (Fig. 5).
- **Its literal channel** is two bytes wide, like the Z80's, and a two-byte self-writer (`B9 51`, `MOV CX,$51B9; PUSH
  CX`) exists.
- **Its block move** (`REP MOVSB`) admits a six-byte overlapping copier, for which the law predicts the payload in
  advance; the census will verify this construction under the final contract (Methods).
- **Validation:** its Zilion core passes 3,137,000 hardware-recorded instruction tests without an unexplained
  divergence under coherent instruction fetch [/Users/neo/repos/zilion-wt-8086/test/baselines/8086-architectural.md].

We register route and heredity predictions separately, from a validated census, before any 8086 soup is run. A wrong
route would refute the first prediction without touching the law; an eligible evolved copier that loses a predicted
payload position would refute the law [PENDING: census, registration and results] (Methods).

### Discussion

Spontaneous replicators differ in what they pass on, and in the Z80 soups this difference has a mechanism. The overlap
of a copy with itself decides which parental bytes reach descendants. The displacement setting that overlap is itself
inherited, so one copying rule controls how much else is inherited, and the environment decides which rules prevail.

The results separate two thresholds that are usually treated as one.
- **Closure:** the offspring no longer depends on the environment it overwrites.
- **Heredity of variation:** the offspring preserves differences in its parent.

The first replicators in benign Z80 soups cross neither. Regenerators cross the first but not the second. Only
transmitters cross both. Woese's Darwinian threshold, the transition to stable vertical descent, corresponds to the
first in our system, and Darwinian evolution in the strict sense needs the second as well [Woese 2002]. We do not
equate a closed program with a biological individual, or claim a universal order. The 6502 begins closed, and lethal
chemistry lets the Z80 skip its open phase. But the separation is general: a population can achieve stable
reproduction, even genealogical continuity, while lacking the heredity that selection requires.

The same separation applies wherever replication is studied without genes.
- **Repeated recovery of a composition** in a chemical or prebiotic system could reflect transmission, reconstruction
  or recapture from the environment. Only an introduced difference, followed through reproduction, distinguishes them.
- **The mechanism we describe** is not a claim that chemistry uses a displacement byte. It is a fully inspectable case
  in which the question has an exact answer: which alternatives does a reproductive process preserve, and which does
  its own operation remove?

Two limits define what comes next.
- **Capacity is not function.** The payload of these transmitters is copied but never read for any other purpose. An
  inherited difference can matter to selection only if something reads it in a second way, as von Neumann's
  description is both copied and interpreted. Whether such dual reading arises spontaneously is the next question
  these soups can answer.
- **The ecology is incomplete.** We know that lethal chemistry favours transmitters, but not yet why.

Zilion makes both questions experimental. Together with the prospective 8086 test, it provides what has been
missing: a way to ask of any spontaneous replicator, on any processor, not only whether it reproduces but what its
reproduction preserves.

## Figure legends

<!-- Legends specify the v5 display plan. Existing figure files have not been rebuilt. -->

**Figure 1 | Zilion makes reproduction an experiment.** **a**, Encounter schematic: focal and partner tapes, reset state, bounded execution, writeback and descendant sampling. Supplied boundaries and execution resources are distinguished from encoded machinery. This is a contract diagram, not a data panel; source: `algocell_exp/shader/` and Zilion `docs/ARCHITECTURE.md`. **b**, Byte-level pusher, regenerator and transmitter traces, distinguishing instruction starts, all instruction-stream fetches, data reads and writes. Data: `results/concept/traces.json` and `results/crystal/roles_*.csv`, `results/crystal/provenance_*.csv`; historical and exploratory trace labels remain visible. **c**, Representative world trajectory linked to isolation and branching assays; the world illustrates sampling, not a median history. Data: `runs/video/video_L16_st128_k4_s2002.jsonl`, `results/video/FIG2_NUMBERS.md`, and `results/branching/branching_summary.csv`. **d**, Validation coverage from instruction semantics through encounters and assays to ecological reruns. Data: `results/impact/outcome.csv`, `results/impact/validation.csv`, `results/tier2/T2_COMPONENTS.csv` and the absolute 8086 baseline cited in Methods. Unmeasured levels remain blank or PENDING; vector totals do not stand for ecological replication.

**Figure 2 | Survival is not retention.** **a**, Registered branching trajectories through generation eight, showing unmutated-population survival separately from mutant-population carriage and thresholded sites. The pusher16 anchor is 48/64 surviving controls with no scored site, versus nine sites in genome20; X1-2 remains labelled failed. [results/branching/branching_summary.csv, gen=8; results/branching/SUPP_TABLE.md] Data: `results/branching/branching_summary.csv`, `results/branching/SUPP_TABLE.md`; the C1 comparison uses `results/tier2/scored/Branching/branching_summary.csv`. **b**, Full retention maps distinguish lost lineages, surviving erasers and surviving carriers, with fetched positions and rare working core alleles marked. Data: `results/serial_retention/sr_mutants.csv.gz`, `results/serial_retention/sr_sites.csv`, `results/serial_retention/sr_summary.csv`; branching alleles from `results/branching/branching_mutants.csv.gz`. **c**, Complete regenerator erasure trace, following the introduced parental byte to its overwrite and the descendant’s subsequent reproduction. Source: exploratory `results/crystal/probes_ldir32.csv`, `provenance_ldir32.csv`, `roles_ldir32.csv`; [fill: archived full ordered trace chosen for this panel]. **d**, Parent-by-environment survival and retention, with extinct combinations distinguished from surviving erasers. Data: `results/pe/pe_matrix.csv`. Points and intervals must retain their actual unit—lineage, branching population or world—and soup-background survival is explicitly annotated.

**Figure 3 | An inherited displacement controls payload.** **a**, Theorem schematic comparing overlapping prefix regeneration with whole-parent copying; causal source coordinates and protected core are shown separately from repeated output positions. **b**, Predicted free payload versus sampled unexecuted-site score, with additional fetched-site scores displayed separately. Evolved Z80 cells, constructed 6502 controls and PENDING evolved 8086 measurements use distinct symbols. Data: `results/offset/offset_cells.csv`, exploratory `results/crystal/cells.csv`, `results/crystal/cell_exceptions.csv`, and `results/census6502/table_closers_law.csv`. World-clustered summaries accompany cell distributions; exclusions and the gate-related deficit are shown. **c**, Working offset alleles, including exact intervention and offspring sequences. Data: `results/serial_retention/sr_mutants.csv.gz` and exploratory `results/crystal/alleles_ldir32.csv`, `results/crystal/alleles_genome20_s6003.csv` and exact first-offspring records in `results/crystal/allele_encounters_ldir32.csv`, `results/crystal/allele_encounters_genome20_s6003.csv`. The aligned-offspring dial is distinguished from serial marker retention. Reciprocal edits, full descendant haplotypes and direct descendant capacity: PENDING [fill: generated targeted-audit table]. **d**, Joint-payload challenge and controls for insufficient coverage, phase and nondividing displacement. [fill: generated joint-intervention table and frozen eligibility manifest]. Existing targeted single-coordinate examples are sourced to `results/crystal/offset_examples.csv` and are not relabelled joint tests. Constructed REP controls and evolved 8086 tests remain separately PENDING.

**Figure 4 | Environments sort reproductive designs.** **a**, Matched benign/lethal trajectories at L=32 [results/tier2/scored/E/offset_switch.csv], with both invasion directions and equal starts, showing transmitter share among core carriers beside absolute core-bearing abundance. **b**, World-level endpoint distributions, distinct tails and core-loss outcomes, preserving failed-core worlds. Data for **a,b**: historical `results/offset/offset_switch.csv` and C1 `results/tier2/scored/E/offset_switch.csv`, with the corresponding archived run trajectories; displayed rounding follows the generated reports. **c**, Lethality dial: first-open frequency, transmitter-majority frequency and the registered confinement timing. Data: `results/dial/dial_worlds.csv`, `results/tier2/T2_DZ.csv`, `results/tier2/T2_COMPONENTS.csv`. Historical DZ2/DZ3 failures and C1 passes appear side by side; the remaining timing reversal is retained. **d**, Encounter outcomes with core-free intruders and a diagram separating measured interventions from unresolved links to equilibrium. Data: `results/offset/OFFSET_ENCOUNTER.md` and `results/tier2/scored/E/OFFSET_ENCOUNTER.md`, generated outcome tables. Rejected toxin, backup-core and mutation-only mechanisms receive no fitted explanatory curve. Trajectory summaries use worlds as replicates; cell and tail counts are nested observations.

**Figure 5 | Different beginnings and prospective tests.** **a**, Census–registration–population chronology for 6502 and 8086; the literal theorem is retrospective for 6502. Sources: `results/census6502/CENSUS.md`, `REVISION_PREREG.md`, section S1, `results/s6502/CORRECTION.md`; 8086 census and registration PENDING. **b**, World-level first-detected route distributions, including censoring and the modal-detection threshold. Data: `results/stageG/stageG/stage_g_runs.csv`, corrected `results/s6502/v2/s1_worlds.csv`; 8086 PENDING [fill: frozen cohort route table]. The descriptive confinement result does not replace failed 6502 P1. **c**, Proposed 8086 default versus controlled pointer-initialization contrast, conditional on the validated census. Literal, memory-fed open and confined routes are separate; inflow is measured separately from fetches. All data, effect threshold and verdict: PENDING [fill: validated census rationale, registration hash and arm-level results]. **d**, Evidence matrix for route, confinement, context dependence and persistent variation, including BFF dissociations. Data: `results/exectrace/per_replicator.csv`, `results/biology/individuality/per_replicator.csv`, `results/bff_dials/d2_bandwidth.csv`, `results/serial_retention/sr_summary.csv`, and corrected 6502 scores. Missing evolved 6502 retention and all 8086 population outcomes remain explicitly missing; the 8086 copying-law slot belongs in Fig. 3.

## Methods

### Execution and the semantic contract

Zilion supplies instruction cores, a shared world interface, an execution engine and machine manifests. Population pairing, mutation, genealogy and scientific classifications remain research code. A manifest binds core identity, semantic version, compatibility, chip variant, generated shader and host hashes to address mapping, initial state, budgets, traps and observations. We do not infer the executed semantics from a package name. The implementation described here is the architecture in `/Users/neo/repos/zilion/docs/ARCHITECTURE.md`; a final release archive and independently runnable encounter benchmark remain to be deposited. [fill: release DOI, frozen manifest catalogue and benchmark table]

Architectural address reduction precedes mapping into the pair ring. This ordering matters when the ring does not divide the architectural address space: crossing the machine’s wrap boundary need not advance to the adjacent ring position. The historical Z80 resets registers and flags to zero except for the stack pointer, initialized to the largest architectural address aliasing the partner’s last byte. This convention specifies the pointer, not necessarily the first written byte. Alternate reset and stack-phase experiments retain their own contracts. Opcode, prefix, operand and displacement fetches are recorded separately from non-fetch data reads and writes; an instruction-start map alone does not identify the fetched set.

Execution budgets count defined work units, not physical clock cycles. Repeating Z80 block moves consume budget on each repetition. Prefix handling, ablation after decoding, HALT, I/O and undocumented behavior are properties of the named core. Historical Z80 I/O reads return zero and writes have no external effect; there is no external interrupt source. The lethal-tar rule halts on a fetched zero opcode, whereas benign zero executes normally. The 8080 subset suppresses Z80-specific operations within the same emulator and is not an independently validated physical 8080.

The 6502 contract uses NMOS instruction semantics, initially clear accumulator and index registers, status `0x20`, entry at the focal tape, and a first stack push aligned to the last partner byte. Architectural zero-page and stack-page accesses are then mapped into the ring. Faithful BRK vectoring and halt-before-BRK are separate arms; JAM halts and no external IRQ/NMI is supplied. BFF’s published contract halts on unmatched brackets and pointer run-off; other bytes outside its instruction set are no-ops. Its literal-write, pointer-wrapping, bandwidth and bracket-halting variants retain separate contracts. [results/s6502/CORRECTION.md, contract audit; results/bff/runs.csv] [fill: generated cross-machine reset, trap and budget contract table]

### Encounters, sampling and statistics

Historical Z80 worlds contain 20,000 tapes on a 160 × 125 lattice. Each step draws 8,192 focal–neighbor pairs, suppresses conflicting pairs under the archived pairing procedure, executes each active ordered pair, and writes back both halves. The nominal budget is 128 instructions. Background mutation selects bytes for random replacement; its declared intensity counts attempts. [results/review_r2/RUN_TABLE.md] [fill: generated encounter-parameter table for population, lattice and pair-draw counts]

Packed-word mutation writes can collide and lose an update. This issue is shared by the historical Z80 and 6502 kernels; its population effect is unmeasured. We do not describe attempted replacements as a measured number of applied mutations. A new mutation implementation would define a new contract. Pairing collision races also prevent exact reconstruction of historical whole-soup trajectories from seeds alone. Replaying a recorded encounter with its actual initial memory is a different, deterministic guarantee.

The modal event dates a sufficiently abundant detected clone, not the first reproductive material. Its frozen rule tests the most common tapes, requiring share at least 0.5% and a passing culture score; random-cell samples estimate population classes independently. [results/stageG/stageG/stage_g_runs.csv] [fill: generated event-threshold table] Historical registrations call the confinement endpoint `t_closed`; that name is preserved in their ledgers but does not turn confinement into a direct zero-inflow measurement. Non-detection is censored at the observation horizon, not treated as impossibility. World-level event times use the registered Kaplan–Meier procedure; interval estimation and hypothesis tests retain the original world-level units and decision rules. Cells, mutant values and descendants are nested observations, not extra worlds. Exploratory ladder times are first detections on an irregular sampling grid; ties, reversals and censored histories remain visible.

### Reproduction, retention and inflow

For a parent $x$, the culture score averages the gain in best-cyclic-shift similarity to $x$, normalized by the initially unmatched fraction. Partners already passing the copy gate are excluded from the gain calculation. The descendant is then tested against fresh partners. “Heritable” in the historical classification means generation-two score at least 0.3; a qualifying copy has at least 75% similarity. These operational labels do not assert exact sequence inheritance. [fill: generated culture and similarity-threshold table; scorer definitions in `assay.py`, `mutscan.py`]

A single-site scan tests whether a mutant remains reproductively competent and whether its changed byte reaches qualifying offspring at the aligned position. At least half of tested values must pass the frozen conditions for a site to score. Population classes use confined cells with at most two sites as regenerators, three or four as intermediate, and at least five as transmitters. “Unexecuted” in these tables means absent from the union of instruction-stream fetches, including operands; it does not mean unused as data. [fill: generated scan-threshold and population-class definition table; `population_classes.py`]

The transmission condition requires the changed byte in at least half of qualifying offspring. Serial and branching scorers additionally exclude mutant windows already present cyclically in the wild type, because those interventions are not identifiable by the marker. Site denominators therefore use identifiable tested values. Conditional carriage among surviving mutant lineages or populations is compared with conditional carriage among surviving controls. In contrast, the plotted lost, erased and retained fate fractions use all original identifiable lineages or populations. These quantities answer different questions and are not substituted for each other. [results/branching/BRANCHING.md; results/serial_retention/sr_mutants.csv.gz] [fill: generated scorer-definition table for conditional denominators and identifiability]

Serial survival requires each transfer to remain a qualifying copy of its immediate parent. Retention searches for the centered five-byte mutant window anywhere cyclically in the descendant, with accidental occurrence estimated from unmutated controls. A site passes when at least half its tested values exceed the control carriage background by the registered 0.5 margin. The archived panel uses sixteen lineages per mutant, 256 unmutated controls and eight transfers. Short tapes receive all 255 alternative values per position; longer tapes receive thirty-two sampled values. [results/serial_retention/sr_summary.csv; results/serial_retention/REPORT.md] [fill: generated window, background and alternative-sampling parameter table]

Branching X1 instead exposes each member to two fresh partners, retains qualifying offspring and randomly subsamples populations above the cap of thirty-two. Each of thirty-two sampled values per site starts four populations, alongside sixty-four unmutated populations. A population carries the allele when at least half its members carry the window. [results/branching/branching_summary.csv, n_mutants/reps/ctrl] [fill: generated branching, cap and carriage-threshold table] The soup-partner arm pools open-phase snapshots across Stage G worlds; it is not the focal parent’s own neighborhood. Already matching partners can inflate survival or reacquire markers. Cyclic windows also do not establish complete haplotype identity, descendant geometry or unique material ancestry.

For deterministic output $Y=F(x,E)$, holding the parent and other context fixed gives $I(E;Y\mid X=x)=H(Y\mid X=x)$, because $H(Y\mid X=x,E)=0$. This is the partner-inflow identity of Lemma 7. Perfect exact copying implies zero inflow; zero inflow alone does not imply copying or preservation of variation. Closure here means zero inflow over the declared context distribution. Sampled zero entropy is evidence within sampled contexts, not a universal certificate. The partner-entropy assay uses 256 outputs and thus has an eight-bit sample ceiling; undersampling biases it downward. [results/biology/individuality/per_replicator.csv] [fill: generated entropy-sampling parameter table] Parental heredity instead compares distinct parents and follows their distinctions. Branching reproduction is a context-conditioned kernel over populations, with extinction included.

Population scans estimate frequencies by culturing and tracing randomly selected cells, then scanning a subset of heritable cells. Unscanned confined cells inherit the estimated class proportions of scanned confined cells, not a directly observed genotype label. The sampling stage, budget and partner panel accompany each estimate. The offset regrouping reuses these observations; its cells are not a new set of independently evolved worlds. Crystal’s targeted interventions, first-tape scans and offspring-period checks have their own denominators. The sparse ladder analysis does not infer unsampled ancestors, and its genealogy records lack contemporaneous R/T labels. [results/population/classes_snapshots.csv; results/crystal/inputs.csv; results/ladder/inputs.csv; results/ladder/LADDER.md]

### The overlap-copy result

**Theorem; restricted forward-copy family.** On a ring of $P$ cells, perform sequential writes $M[s+d+k]\leftarrow M[s+k]$, with addresses reduced modulo $P$, $0<d<P$, and no interfering operations. For $k<d$, the source has not yet been written and supplies its initial value. Thereafter its latest write occurred $d$ steps earlier. Induction gives original source coordinate $s+(k\bmod d)$ for every copied byte. If $d\mid P$, every write supplies the correct residue of a $d$-periodic tiling. Coverage $K\ge P-d$ fills the remaining ring. With $P=2L$ and $d\mid L$, the partner begins at the same phase as the parent.

The Z80 specialization retains the audited hypotheses of Proposition 8. For `XX 5e ed b0`, $L\le64$, $d=XX\bmod2L$, $4\le d\le L$, and $d\mid L$. The initial one-byte instruction must pass control to address 1, write only at addresses $d$ through $2L-1$, and leave $HL\equiv0$, $256D\equiv0\pmod{2L}$, with $BC=0$ or $BC\ge126$. Address increments must preserve the assumed displacement through the relevant work. These are theorem and contract constants, not population estimates. [manuscript/MAIN_nature.md, audited Proposition 8] [fill: generated proof-contract parameter table]

The unchanged initial byte is then loaded into E, establishing the displacement. Under the nominal budget, 126 block-copy executions remain; their coverage suffices at the stated lengths. The copied core is unchanged when revisited, so execution remains in the block move. The immediate form `1e NN ed b0` supplies the offset directly and leaves 127 block-copy executions. [theorem arithmetic under the same proof contract] Prefix positions outside the core can vary jointly without changing these conditions. The offspring retains that prefix and the same core, so induction across encounters proves serial transmission. Parental positions beyond the prefix are overwritten before supplying independent final information.

Thus, for constrained footprint $c$ wholly inside the protected prefix, the invariant product family has $r=d-c$ free coordinates, $256^r$ possible sequences and $8r$ ideal noiseless payload bits. This is cardinality arithmetic, not measured Shannon capacity, ecological neutrality or unlimited heredity. Working core alleles are additional possibilities outside this fixed-core count. A taken return can divert execution, a store can destroy the offset before it is read, insufficient coverage can leave partner bytes, and nondividing displacement can change the offspring phase or corrupt the core. Reverse `LDDR` designs require another derivation. The population motif census supplies candidates, not automatic eligibility certificates.

### A sufficient provenance certificate

Fix the admissible family, reset and partner contexts, budget, descendant boundary and alignment before measuring retention. Assign symbolic labels to original parent and partner coordinates. Unroll reads and writes in time order: a copied byte inherits the source’s current label, which may already descend from an earlier write. For each proposed payload coordinate, prove that its corresponding aligned offspring coordinate carries that parental label after the final write, without partner substitution. Then establish that this property holds for jointly arbitrary payload assignments and that every admissible offspring remains in the same certified family. Induction proves retention on repeated reproduction.

The guarantee must be nonvacuous: every parent and context covered by the certificate must produce the required viable descendant. Its entry phase and reset convention must permit the same argument at the next transfer. A family may allow an explicitly tracked coordinate permutation, but the extraction rule must preserve distinguishability rather than silently identify different payloads by a favorable cyclic shift. For the overlap family, the protected prefix and dividing displacement supply that invariant alignment. For other geometries it must be established separately.

Noninterference with the required execution is a sufficient way to discharge these obligations; it is not necessary for heredity. Address-changing alleles, successful value-dependent schedules and arithmetic encodings may preserve distinctions outside this pure-copy certificate. A trace at one assignment proves only that execution’s provenance. The exploratory crystal response maps are local intervention diagnostics, not invariant-family certificates. Eligibility must be fixed from code and provenance before retention outcomes; the core footprint must never be inferred by subtracting observed sites from displacement. Complete machine-checked certificates and joint-payload intervention tables remain release obligations. [fill: certificate archive and generated eligibility/exclusion tables]

### Literal channels and their limits

**Theorem; repeated load–push pair.** Let distinct one-byte opcodes load $w$ literal bytes into a shared register and push a fixed permutation of them without altering that register. The stack pointer is separate and advances by $-w$; writes form contiguous blocks with no later interference. Assume the executed stream remains the prescribed repetition of load, literal and push. Execution and output periods are $w+2$ and $w$. A finite matching window permits the Fine–Wilf conclusion only when

\[
N\ge2w+2-\gcd(2,w).
\]

At this corrected bound, odd widths force a constant stream, contradicting distinct opcodes; even widths permit the alternating construction with a compatible output permutation. Ascending memory reverses the order of descending stack blocks, not their internal byte layout. Placement, coverage, phase and bootstrap are separate encounter requirements. [results/literal_channel/CHECK.md, corrected proof; results/literal_channel/pair_layouts.csv]

**Theorem; width-one periodic mixtures.** For primitive input period $d$, deterministic decoding has a recurrent cycle of at most $d$ boundary residues. If it contains both loads and pushes, its emitted length $n_P$ satisfies $0<n_P<t\le d$. An output equal to the periodic input would require $d\mid n_P$, a contradiction. Multiple allowed opcode identities preserve the argument. The degenerate all-push writer supplied an opcode-valued initial register is excluded. This proves an infinite periodic-stream obstruction for the stated single-register grammar, not absence of all finite 6502 writers. [results/literal_channel/CHECK.md; results/literal_channel/minimal_symbolic_search.csv; results/literal_channel/direct_search.csv]

The limits have constructive witnesses. At width three, `(abb)` repeated decodes as load `a[bba]` followed by two pushes and emits a matching shifted stream: the repeated-pair parity rule does not extend to mixtures. At width one, the archived finite-factor witness can match a finite rotated tape under prescribed periodic execution, yet fails zero-partner bootstrap when execution reaches unwritten memory. [results/literal_channel/verification.csv; results/literal_channel/finite_search.csv; results/literal_channel/finite_witness.json; results/literal_channel/6502_finite_harness.json] [fill: generated finite-witness/bootstrap summary table] Z80’s five nontrivial period-two writers likewise include only three literal channels; the remaining forms load from memory. [results/literal_channel/census_audit.csv] None of these results predicts population arrival order by theorem.

### Validation and correction studies

Validation has distinct instruction, encounter, isolated-reproduction and ecological levels. Hardware vectors test covered instruction semantics; independent reference comparisons test their intersection with ring behavior. A CPU implementation derived from the same core is not an independent oracle. Historical Z80 differential tests cover the ring sizes and targeted programs listed in `results/emulator/RING_DIFFTEST.md`; uncovered rings and expected divergences remain explicit. Recorded genealogy encounters replay correctly within their archived scope, but this does not make whole soups deterministic. Frozen representative tapes and counterexamples provide the bridge from instruction tests to reproduction assays.

I1 compared historical C0 compatibility with corrected classic C1 and vendor-specific C2 semantics. C0 uses compatibility `0.2`; C1 and C2 use `0.3`, with C2 selecting `zilog-nmos`. Its generated decision table flags 47 of 690 conditions for C1 and 114 of 690 for C2; each comparison has 407 identical and ten differing frozen assay cases. The registered all-unaffected criterion was not met. Differences include refresh handling, memory-dependent flags and repeating-block flags; C2 additionally models the vendor Q latch. Stochastic-world assays use executor case-index streams, not archived encounter streams. I1 identifies dependence on core semantics, not a causal population effect. [results/impact/README.md, generated decision table; results/impact/decisions.csv; results/impact/assay_identity.csv; results/impact/cores.csv; results/impact/limits.csv]

I1-T2 preserves 19 of 22 claim verdicts while changing four of 36 components. “Preserved” includes negative and no-direction verdicts. DZ2 and DZ3 change as reported in the main text; Stage L’s L2 becomes not applicable at length sixteen and not met at length twenty, while its joint kill still fires. X1’s survival/retention contrast and the lethal competition endpoint stand. Original failures remain alongside C1 rerun outcomes. Fresh nondeterministic populations cannot isolate the effect of the core repair without an appropriate causal design. [results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv]

### The coherent-fetch 8086 contract

The selected 8086 abstraction fetches consumed instruction bytes from current memory and has no prefetch queue. The hardware ledger records 3,137,000 8088 cases: 3,126,917 raw architectural passes and 10,083 expected divergences, with zero unexpected. The core remains a development implementation pending the research freeze. The expected rule covers undefined far-register transfers whose hidden latch state is absent from the fixtures; these forms can affect soup control flow and are not validated away by the rule. Registers, flags, memory, ports, ordered writes, halt/trap and REP completion are compared. Queue state and cycle timing are excluded. [absolute source: /Users/neo/repos/zilion-wt-8086/test/baselines/8086-architectural.md, generated Summary and Expected divergences]

Segment translation and architectural wrapping precede ring mapping; each byte of a word wraps its offset independently. Segments, entry and general registers default to zero, with FLAGS `F002`; DF and TF are fixed, not randomized with pointers. Word pushes decrement SP before low-then-high writes. The first-write phase must be frozen and bootstrap retested: the exploratory zero-SP convention differs from `firstPushAt:lastByte`. REP iterations consume work budget; zero count copies nothing. Decoded operation and source segment persist across iterations, so coherent fetch does not imply repeated decoding. Interrupt vectors alias ring memory; external interrupts are absent. With no device, input returns `FF` per byte and output is discarded. HLT reports a halt unless explicitly ignored; WAIT without a coprocessor is a no-op. Trap-vector policies and hidden-state-dependent undefined forms require explicit treatment in the frozen research manifest. [absolute source: /Users/neo/repos/zilion-wt-8086/docs/I8086.md] [fill: generated frozen 8086 contract table]

### Pending prospective design

The 8086 census, registration and results are PENDING. The charter specifies the sequence, not a completed preregistration. First freeze the machine and encounter tests, including stack phase, PUSH SP, bootstrap, word order, wrap seams, vector aliasing, REP and coherent self-modification. Then census all 256 period-one and 65,536 period-two words, constants and both phases included, at lengths sixteen/thirty-two and budgets 128/512 against zero and frozen random partners. Archive all unknowns, primitive-period duplicates, mechanism classes, coverage and inflow separately. [fill: generated complete census and contract manifests; design choices from JOINT_CHARTER §12]

An exhaustive census requires no unevaluated words. Candidate audits include the literal `B9 51`, indirect-memory `FF 30`, and load–push `AD 50` forms under the final contract; opcode labels alone do not determine their reproductive mechanism. If the validated inventory changes the expected route, the prediction must change before populations are observed. Throughput measurements use fixed inputs and cannot serve as undisclosed population pilots.

Before soups, freeze seeds, cadence, horizon, censoring, exclusions, route thresholds and law eligibility. Retain the comparison protocol’s lattice and encounter intensity, and declare mutation attempts separately from applied updates. The proposed primary and secondary cohorts each contain twenty worlds, respectively at lengths sixteen and thirty-two with budget 128 and a three-million-step horizon. Add twenty controlled pointer-initialization worlds only if the census establishes the proposed selective effect. The conditional primary route criterion is at least 80% open first copiers with at least ten detections; fewer detections leave route frequency uninformative. One-pass-memory occurrence and the pointer contrast require their own numerical thresholds before launch. Secondary results cannot rescue primary failure. [fill: generated frozen 8086 registration/design table; all numbers are proposed design choices, not results]

For constructed or evolved `BF dLo dHi 49 F3 A4`, the conditional prediction $d-6$ requires zero source/destination segments and source index, forward direction, $6\le d\le L$, $d\mid L$, sufficient writes, an intact copied core and no interference. Evolved candidates must be selected by a frozen world/snapshot rule including random cells. Predict source coordinates before testing every alternative payload value, joint assignments, erased-coordinate controls, core inheritance and serial passage through generation eight. [theorem/construction prediction] [fill: generated eligibility, intervention and serial-protocol tables] Score causal transmission separately from the whole-tape gate. A reproducible provenance-resolved mismatch in eligible viable descendants demands diagnosis; approximate site-score agreement is not an exact-law pass. Constructed controls cannot replace absent evolved cases.

Constructed REP controls start from $d$-periodic parents with the core at the offspring entry phase. The size predictions are conditional arithmetic: for $L=16$, offsets eight and sixteen yield two and ten free coordinates; for $L=32$, offsets eight, sixteen and thirty-two yield two, ten and twenty-six. [theorem, $r=d-6$] [fill: generated constructed-control design table] Serial challenges use independent fresh partners, follow complete core haplotypes and distinguish causal inheritance from partner reacquisition. All exclusions and world-level coverage of qualifying architectures are reported, including a possible empty eligible set.

### Registration chronology and availability

The SI chronology separates discovery stages, prospectively registered tests, amendments, post hoc analyses, constructions and theorems. X1-2 remains failed. Genealogy definitions were settled on discovery records before N6; its failed component remains failed. X3’s first archive existed before its registration, unlike its prospective replication. The validated 6502 census preceded S1, whereas the literal theorem is retrospective. Corrected S1 scoring retains P1 failure and untested evolved-offset P3, and discloses registered-versus-executed partner-count and partner-source discrepancies. Crystal, ladder and local copy maps are exploratory throughout. The 8086 registration is PENDING. [REVISION_PREREG.md; results/lod_n6/N6_SCORE.md; results/s6502/CORRECTION.md; results/crystal/CRYSTAL.md; results/ladder/LADDER.md]

The release must bind each figure to raw inputs, contract, scorer and generated table; include deterministic encounter benchmarks, frozen representatives and exclusions, proofs, divergence ledgers and figure commands. Historical and C1 archives remain separate. [fill: immutable data/code DOI, release manifest and v5 figure-reproduction commands]

## Extended Data and SI map

The following are allocations for v5, not claims that new displays or SI files have already been rendered.

| Display | Content and source family |
|---|---|
| Extended Data 1 | Validation coverage and exclusions; I1 and I1-T2 component-by-component ledgers, historical/C1 identities, mutation disclosure, and the hardware-versus-encounter distinction. `results/emulator/`, `results/impact/`, `results/tier2/`, Zilion baselines. |
| Extended Data 2 | Complete serial and branching maps, surviving erasers, rare core alleles, background carriage and threshold sensitivity. `results/serial_retention/`, `results/branching/`, `results/pe/`. |
| Extended Data 3 | Offset-allele dial, serial core variants, payload-versus-total decomposition, deficit diagnosis and minimal-period counterexample. `results/offset/`, `results/crystal/`; direct descendant-capacity audit PENDING. |
| Extended Data 4 | Reset dependence, random-cell sampling, mode-versus-population disagreement, marker failures, and exploratory R/T detection order. `results/r4/`, `results/population/`, `results/marker_check/`, `results/ladder/`. |
| Extended Data 5 | Full R/T competitions, absolute core abundance, core loss, tail diversity, encounter components and rejected ecological mechanisms. `results/offset/`, C1 `results/tier2/scored/E/`, N1/N2/N4 records. |
| Extended Data 6 | Validated finite censuses, near-copiers, constructed law controls, corrected S1 endpoints and sensitivities; pending 8086 census/contract sensitivities. `results/census6502/`, `results/s6502/`, `results/literal_channel/`. |
| Extended Data 7 | BFF literal, bandwidth, pointer and halting controls, including open-fetch/zero-inflow dissociations and failed registered components. `results/bff/`, `results/bff_dials/`, BFF inflow tables. |
| Extended Data 8 | One genealogy synthesis distinguishing discovery definitions from N6 confirmation, first founders from later conversions and functional donor definitions; retain failed N6 component. `results/lod/`, `results/lod_n6/`. |

The SI holds full proofs and computed provenance certificates; every genotype, exclusion and assay definition; original registrations and dated amendments; original and updated verdicts; failed hypotheses; the early length, ablation and padding atlas; genealogy definitions and correction history; raw-data dictionaries; manifests, vector coverage and divergence ledgers; performance measurements; and reproducible figure commands. Detection benchmarks remain archived technical controls rather than a separate biological claim. Withdrawn interpretations are documented as corrections, not retained as premises. No universal succession, neutral-manifold, general parity, selection-for-evolvability or open-endedness claim is assigned to a display.
