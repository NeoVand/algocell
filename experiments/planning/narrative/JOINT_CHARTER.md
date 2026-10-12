# Joint charter: what reproduction preserves

11 October 2026. Astra's proposed joint charter after reading ASTRA_R1, CLAUDE_R1, CLAUDE_R1b and CLAUDE_R2, auditing the updated record, and checking the relevant literature. This consolidates agreement and identifies unresolved disagreements; it does not imply Claude has approved this text. The user's decisions are binding: **Zilion is introduced in this paper, with no separate tool paper; the 8086 is the third processor.** No new experiments were run for this charter.

## 1. Editorial decision

**Thesis, one sentence:** In spontaneously evolved digital replicators, the copying mechanism determines which variations survive reproduction; in a recurring architecture, an inherited displacement allele sets the number of freely transmissible payload positions, and the environment changes the prevalence of these reproductive designs.

**The story.** We built Zilion to make the mechanics of reproduction experimentally inspectable. It exposed a distinction that abundance and copying fidelity obscure: a population can reproduce successfully while erasing the variations introduced into it. Branching-transfer experiments separate that erasure from simple lineage extinction. An overlapping copy explains the difference between restoring a pattern and preserving a sequence, and predicts the locations of freely transmissible bytes. The displacement controlling that transformation is itself inherited: reproductive machinery can change what subsequent reproduction preserves. Environmental interventions alter the abundance of these designs, without demonstrating selection for their future adaptive potential. The 6502 changes the route into replication; a prospectively registered 8086 experiment will separately test route predictions and transfer of the copying law. The paper's opinion is that **an account of the origin of replication is incomplete until it explains the inheritance of differences**. Its contribution is the evolved mechanism, quantitative intervention predictions and comparative experimental instrument—not priority for that conceptual distinction.

**The central explanatory claim is strong enough without Claude's proposed universal necessity.** The most useful surprise from this cross-critique is that evidence for his proposal B is a counterexample to the ordinary wording of his proposal A.

## 2. Hard critique of proposal A

### “A replicator can pass on only what its reproduction does not depend on”

**Verdict: false as stated; a useful sufficient condition has been promoted into a necessary condition.**

The offset byte changes the addresses used by reproduction and can nevertheless be inherited. This is not just a hypothetical objection. The archived serial assay includes mutations of an evolved regenerator's offset that remain detectable through eight transfers:

| Evolved parent and intervention | Mechanistic prediction | Archived generation-eight result |
|---|---|---|
| `ldir32 = (04 5e ed b0)^8`; position 0 changed to `08` | On the 64-byte pair ring, displacement changes from 4 to 8; fixed-core free payload changes from 0 to 4 bytes. | `alive=1`, `retained=1`, background 0. |
| Same parent; position 0 changed to `60` | `0x60 mod 64 = 32`; fixed-core free payload becomes 28 bytes. | `alive=1`, `retained=1`, background 0. |
| `ldir20 = (1e 04 ed b0)^5`; position 1 changed to `14` | Immediate displacement changes from 4 to 20; fixed-core free payload becomes 16 bytes. | `alive=1`, `retained=1`, background 0. |

These are archived mutant interventions on evolved parents, not observations of spontaneous offset sweeps. The assay searches for the centered five-byte mutant window at any cyclic position; lineage survival separately uses at least 75% best-cyclic-shift similarity. It does not independently remeasure the complete offspring copying kernel or its new payload capacity. Nevertheless, it directly supports inheritance of working offset alleles. Sources: [parent definitions](../../serial_retention.py), [raw serial-mutant table](../../results/serial_retention/sr_mutants.csv.gz), keys `(ldir32, gen=8, pos=0, val=8 or 96)` and `(ldir20, gen=8, pos=1, val=20)`; [assay report](../../results/serial_retention/REPORT.md).

Claude's formal version—“heritable for every value if and only if control flow, addresses and stopping are unchanged”—also fails. A copied byte could choose the forward or reverse order of a successful copy, or select one of 256 delays before a sufficiently budgeted copier runs. Every value can be inherited while the execution changes. These are logical counterexamples, not claimed discoveries in our soups. Restricting the slogan to *arbitrary* variation therefore does not rescue the iff.

The defensible sufficient certificate is narrower: a declared set of values is copied through a causal path to the offspring, does not interfere with the conditions needed for that copy, is not overwritten later, and leaves descendants in the same admissible family under the stated contexts. “Unexecuted” alone is insufficient: an unexecuted byte can supply an address, be erased, or remain uncopied. “Fetched” alone is insufficient to establish functional dependence: fetches, operand use, influence on a trace, influence on successful reproduction and influence on transmission are different things. Lemma 11 cannot turn X1 into a universal zero-heredity theorem.

The record already supplies additional limits:

- In the C1 Stage L rerun, seed 6010 at L=20 increases from two to three scored sites, **none unexecuted**. That is a sampled site statistic, not proof of all-256-value neutrality, but it rejects a blanket identification of heredity with unexecuted bytes. [I1-T2](../../results/tier2/T2.md)
- Pusher64 has zero generation-eight sites by X1's threshold, yet 5/2,048 tested mutant types retain the allele with uniform partners and 18/2,048 with soup partners. “Zero sites” and “no mutation survives” are different claims. [X1 supplementary table](../../results/branching/SUPP_TABLE.md)
- The constructed 8080-subset closer retains 12 payload positions **and** seven or eight scored core positions, depending on phase. Even total transmissible-site count need not equal payload length. [Serial report](../../results/serial_retention/REPORT.md)

**Hard to vary?** The literal slogan is easy to vary: “depends on” can retreat from a trace change to a viability change to whatever prevents inheritance. At the last stop it is nearly tautological. The overwrite mechanism is hard to vary: it specifies which original byte supplies each descendant byte and predicts exact interventions before their outcomes.

**What falsifies what?** The ordinary necessity is already contradicted by inherited offset variants. The all-value iff is logically refuted by a value-dependent successful copier. A correctly stated sufficient lemma would be refuted by an admissible, copied, noninterfering value that nonetheless disappears while its guaranteed descendants remain viable. The substantive overlap prediction fails if a qualifying copier loses a predicted free payload coordinate or causally preserves an excluded parental coordinate. A value reintroduced by a partner is not such a preservation counterexample.

### “Capacity is the dimension of a neutral manifold”

**Verdict: retain a projection/identity schematic; reject this as a general dynamical theorem.**

There is generally no single context-free reproduction map. The object is a kernel `K_e(y | x)` for a specified partner/reset distribution, budget and descendant rule. In branching assays, `y` is a population, with extinction explicitly included. Failure to recover a mutant can mean erasure, death or conversion to another reproductive design; it does not establish attraction to the original genotype.

A finite byte space has no intrinsic smooth manifold dimension. Two isolated stable copying forms can transmit a binary distinction while each is a zero-dimensional point. An inherited permutation or alternating cycle need not be a fixed point. Most decisively, our “zero-site” regenerators possess heritable working core alleles. They cannot be described as unique point attractors on that basis.

There is a precise, valuable restricted statement: hold a qualifying core fixed and prove that **jointly arbitrary** values at `r` independent payload coordinates reproduce indefinitely. The resulting invariant product family contains `256^r` sequences, corresponding to `8r` ideal noiseless payload bits. For the specified overlap family, `r=d−c`. This is a cardinality result under the model, not a measured Shannon capacity of the ecological soup. Separate single-site tests do not prove joint independence; the copy proof can, and joint interventions should check its application. Fitness neutrality in a population is also distinct from leaving the isolated copying trace unchanged.

Use a figure showing “different inputs collapse to one repeated pattern” beside “payload distinctions remain distinguishable.” Do not write “the origin of heredity is the birth of a neutral manifold.” The origin, persistence and ecological establishment of the relevant families have not been reduced to that bifurcation.

## 3. Hard critique of proposal B

### “The capacity for heredity is itself a heritable byte, shifted by the environment”

**Verdict: the most promising sharpening, but say precisely what the byte controls.**

In qualifying Z80 cores, this is a real mechanism. For `XX 5e ed b0`, the first byte executes as an instruction and is subsequently read by `LD E,(HL)` to set displacement. Under Proposition 8's hypotheses it belongs to the protected prefix, is copied into the offspring core, and sets the same displacement when that offspring executes. In `1e NN ed b0`, the immediate operand `NN` performs the corresponding role. The raw serial examples above establish that some changes of this parameter survive repeated reproduction. This is stronger evidence than simply recognizing an offset motif in a snapshot. [Proposition 8](../../manuscript/MAIN_nature.md)

But capacity is not literally one byte. It also depends on core footprint, source and destination initialization, direction, coverage, ring geometry, stopping and the contexts in which reproduction is viable. Many offset-byte values are lethal or fail to copy. In the self-reading form, changing the byte changes an instruction as well as the displacement. The byte is a heritable **control parameter of a reproductive mechanism**, with a restricted set of working alleles; it is not itself a universally free payload coordinate.

Nor is this the “dominant copier family” without a qualified denominator. The four-byte forms occur in **2,347/15,655** sampled confined heritable cells. They are a recurring, especially tractable architecture, not the whole ecology. [Core-family census](../../results/offset/CORE_FAMILIES.md)

The population law also needs a better comparison than the headline 97.7% agreement. The published table reports total scored sites equal to `d−4` in 2,289/2,343 cells with dividing offsets. A read-only regrouping of its [raw rows](../../results/offset/offset_cells.csv) finds that **2,342/2,343 have `n_unexec=d−4`**. In all 53 excess-total-site cases, unexecuted sites equal the predicted count and the total exceeds it by one or two. That is consistent with additional executed-site heredity, not a failure of a theorem about fixed-core payload. Confirm positions and admissibility before promoting this new post hoc regrouping to a manuscript result.

The single deficit is Stage I, L=16, seed 4004, tape `c8 5e ed b0 e9 c2 ad 00 9a fd d3 f5 be 3e 31 cc`, with predicted `d=8`, four payload positions, and zero scored sites. It is not initially an eight-byte tiling. Tiling its prefix preserves only 8/16 original positions, suggesting that the assay's 75% copying gate, rather than payload erasure, rejects it. This is a diagnosis to test, not a resolved exception. The population scan samples 16 alternative values per site and 16 partners; it does not establish universal transmission of all values. [Scan implementation](../../population_classes.py), [mutation scorer](../../mutscan.py), [original law summary](../../results/offset/COPY_PERIOD_LAW.md)

**The ecological part is supported at the design/class level.** In the C1 L=32 mutation-on 50:50 experiments, benign and lethal environments end near median transmitter shares 0.05 and 0.76 respectively; the lethal registered endpoint remains met. DZ3 now has transmitter-majority populations in 8/10 p=1 worlds. But the seeded R and T recipes differ in tail sequence as well as displacement, and class frequencies are not lineage-resolved offset sweeps. We have evidence for environmental sorting of reproductive designs, not a complete causal account of selection on `d` alone. [C1 offset results](../../results/tier2/scored/E/REPORT.md), [seed recipes](../../offset_switch.py), [T2 ledger](../../results/tier2/T2.md)

**Hard to vary?** Strong at the local level: reciprocal offset edits, followed by exact trace and retention assays, should change a specified set of coordinates without fitting a new parameter. Weaker at the ecological level: rejected toxin, backup-core and isolated mutation-flow explanations leave the equilibrium mechanism incomplete. The corrected DZ2 verdict permits one reversal; it does not establish a strictly monotone law or validate the proposed closure-window mechanism.

**Falsifiers.** Working offset alleles that are systematically reset or sourced from partners rather than inherited would defeat the hereditary-parameter claim in those lineages. Matched offset interventions that fail to change payload retention as predicted would defeat its mechanistic application. A held-out environmental switch that fails its registered design-frequency prediction would defeat that ecological prediction. Failure to evolve a useful payload would not falsify storage capacity; it would prevent a claim that the capacity has supported cumulative adaptation.

**Adopt this sentence:** “Within this copying family, an inherited displacement allele controls how much arbitrary payload survives reproduction; environmental conditions alter the abundance of the resulting designs.”

## 4. What is new, against the literature

| Prior work | What it already owns; what this paper can add |
|---|---|
| [von Neumann, *Theory of Self-Reproducing Automata*](https://fab.cba.mit.edu/classes/MAS.865/topics/self_replication/VonNeumann.pdf) | Interpreting a description to construct and copying that description are established distinctions. The description may influence construction while being inherited. Our result is not a proof that reproductive dynamics must be independent of heredity. Add spontaneously evolved copying architectures and intervention-resolved retention. |
| [Wagner–Altenberg, 1996](https://dynamics.org/Altenberg/PAPERS/CAEE/) | Evolution of the properties that generate useful variation is established territory. Free payload capacity is a possible ingredient, not equivalent to evolvability in that sense. An inherited single-byte modifier is a concrete realization, not the discovery that evolvability can evolve. |
| [Schuster et al., 1994](https://doi.org/10.1098/rspb.1994.0040); [Greenbury–Ahnert, 2015](https://doi.org/10.1098/rsif.2015.0724) | Neutral sequence sets, and counting constrained versus unconstrained coordinates, have close precedents. Add a derivation from overwrite dynamics and a test in evolved replicators. Neutrality alone is not a reproductive channel. |
| [Vasas–Szathmáry–Santos, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2824406/) | Reproducing/self-maintaining composition can fail to preserve selectable differences away from its built-in behavior. This distinction is central prior art, not ours. Add a byte-resolved mechanism and environmental interventions. |
| [Vasas et al., *Evolution before genes*, 2012](https://doi.org/10.1186/1745-6150-7-1) | Autocatalytic cores and compartments can support limited heredity and adaptation. Do not cite the 2010 result as a universal prohibition on autocatalytic evolution without uninterpreted coordinates. |
| [Szathmáry–Maynard Smith, 1997](https://doi.org/10.1006/jtbi.1996.0389); [Zachar–Szathmáry, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2850328/) | Limited/unlimited heredity concerns the repertoire of reproducible alternatives, with relevant population context. Zero broadly mutable sites does not mean zero alternatives. Finite tape assays do not establish unlimited or open-ended heredity. |
| [Agüera y Arcas et al., 2024](https://arxiv.org/html/2406.19108v1); [Cicala et al., 2026](https://arxiv.org/html/2607.09211v2) | Spontaneous digital replication, stack-to-block-copy succession, compact cores and environmental shaping of replication architecture are prior art. Credit the earlier partner-offset annotation explicitly. Add persistent-variation assays, the overlap law's explanatory use, parameter inheritance, and prospective cross-machine tests. |

The overlap operation itself is familiar programming behavior. The novelty claim is **what it explains about experimentally evolved heredity**, supported by a mechanism-to-lineage-to-ecology chain. This is a targeted primary-source check, not proof of bibliographic priority. No “first” claim is needed.

## 5. Title shortlist, bold to safe

1. **Reproduction can erase its own variation** — strongest provocative title that the measured contrast supports.
2. **The copying mechanism sets the capacity for heredity** — recommended; the abstract immediately defines capacity and scope.
3. **An inherited copying rule controls what can be inherited** — foregrounds the parameter result; use only with the direct descendant-capacity audit.
4. **How digital replicators preserve and erase variation** — clear and broad.
5. **Copying overlap and heritable variation in digital replicators** — safest, least reach.

Retire “replicators before individuals,” “the birth of the self” and “a Darwinian threshold” as titles. Individuality remains a question motivated by the results, not their established common invariant.

## 6. Abstract — current-evidence draft, approximately 150 words

Self-replication can preserve a form while erasing the variations introduced into it. We introduce Zilion, a validated execution substrate for studying reproduction in interacting populations of machine-code programs. In Z80 populations, branching lineages persisted while losing the tested mutations, whereas other evolved copiers transmitted sequence variation across generations. An overlapping-copy mechanism explains this contrast: for a specified architecture, displacement minus the fixed core length gives the number of independently transmissible payload positions. Mutations of the displacement parameter itself survived serial reproduction, showing how an inherited copying rule can change what else is inherited. Changing the execution environment shifted the prevalence of reproductive designs with different hereditary capacities. A census-guided experiment on the 6502 found a different route into replication, separating that route from the copying law. These results make heredity an experimentally measurable property of reproduction and identify a concrete mechanism through which reproducing systems can acquire the capacity to preserve new differences.

**Submission update:** replace the 6502 scope sentence with the actual three-machine result after the registered 8086 analysis. Do not prewrite a successful 8086 outcome. This draft's “independently transmissible” statement refers to the theorem's jointly free payload, not the population site's threshold statistic.

## 7. Five main figures

### Figure 1 — Zilion makes reproduction an experiment

- **a. The encounter and its contract.** Two tapes, reset state, execution, writeback and descendant sampling. Make supplied cell boundaries and execution resources visible; show which components actually evolve.
- **b. Three mechanisms in bytes.** A pusher, a regenerator and a transmitter, with instruction fetches, source reads and writes distinguished. Explain why a successful copying assay alone cannot identify a hereditary mechanism.
- **c. From world to testable lineage.** One representative population trace plus the isolation/branching assay. Mark persistence, confinement, partner dependence and retention as separate measurements.
- **d. Instrument validation ladder.** Instruction vectors → ring encounters → reproductive assays → ecological reruns. Show bounded coverage and the historical/C1 distinction; no software architecture poster or throughput trophy.

### Figure 2 — Survival is not retention

- **a. Branching controls and interventions through generation eight.** Show survival and allele retention with their own denominators. Anchor the result with pusher16's 48/64 surviving unmutated populations and zero scored sites, versus the transmitter's nine sites. Preserve the original registration's failed X1-2 label.
- **b. Full retention maps for representative designs.** Separate dead lineages, surviving lineages that erase the allele, and surviving carriers; annotate executed sites. Show rare retained core variants instead of making “zero” visually absolute.
- **c. One complete erasure trace.** Follow an introduced byte through a reproducing regenerator; make erasure a visible operation rather than an entropy metaphor.
- **d. Parent-by-environment result.** Distinguish extinction from survival without payload retention. This establishes that an assay failure caused by ecology is not automatically a loss of the underlying copying capability.

### Figure 3 — An inherited displacement controls the payload

- **a. Two copy geometries on the same byte diagram.** Derive provenance under forward overlap, protect the core explicitly, and distinguish independent source coordinates from repeated descendants.
- **b. Predicted versus measured free payload.** Z80 evolved cells, constructed 6502 controls, and prospectively tested evolved 8086 cells, using distinct symbols. Show eligibility, excluded cases and residuals; do not plot total scored sites against a non-core theorem without decomposition. Cluster summaries by world, not by cell as independent replicates.
- **c. The inherited switch.** Exact parent/offspring sequences for working displacement alleles plus reciprocal edits and descendant capacity measurements. Use the existing serial evidence now; complete the targeted assay before submission.
- **d. Joint payload challenge and boundaries.** Retain combinations of new tags under qualifying geometry; show a nondividing or insufficient-budget control where the simple formula does not apply. No successful evolved 8086 cases means an explicitly untested slot, not replacement by constructed cases under the same label.

### Figure 4 — The environment sorts hereditary designs

- **a. Matched L=32 benign/lethal trajectories.** Show both invasion directions and 50:50 starts where available, with transmitter share and absolute core-bearing abundance.
- **b. Final distributions and tail diversity.** Report world-level distributions, failed-core worlds and the distinction between relative T share and absolute persistence.
- **c. C1 lethality dial.** Open-first frequency and transmitter-majority frequency; a small inset reports closure timing, including its remaining reversal. Original and C1 verdicts are distinguished.
- **d. What is explained and what remains open.** Empirical encounter outcomes with third-party intruders, alongside the causal links established by intervention. The environment-to-equilibrium mechanism is explicitly incomplete; no successful-looking curve from the rejected mutation-only model.

### Figure 5 — Different beginnings, prospectively tested mechanisms

- **a. Census → registration → population run.** Separate chronology for 6502 and 8086. The corrected literal theorem is retrospective for 6502; 8086 route and law predictions must precede its soups.
- **b. First detected reproductive routes.** Z80, 6502 and 8086 world-level distributions with censoring and sampling limits. Preserve the 6502 registered P1 failure alongside the descriptive 20/20 confinement result.
- **c. The 8086 discriminating intervention.** Default versus controlled pointer-register initialization, if the validated census supports the contrast. Distinguish literal pushers, memory-fed open copiers and confined loops; include partner inflow separately. A persistent open copier is a result, not an inconvenient unfinished transition.
- **d. The reach of the explanation.** A compact matrix of route, confinement, context dependence and persistent variation, including BFF's decisive dissociations. Missing retention measurements remain marked missing. The 8086 law data themselves belong in Figure 3, so this figure tests route rather than repeating the law plot.

## 8. Main-text outline and word budget

Main prose target: **3,500 words**, excluding the approximately 150-word abstract, legends, references and Methods.

| Section | Words | Job |
|---|---:|---|
| Introduction: what a copier passes on | 350 | Credit the conceptual precedents; state the unresolved mechanistic question and the paper's opinion. |
| Zilion and the experimental contract | 450 | Introduce the instrument here; define the encounter and the four observables. |
| Reproducing lineages can erase tested variation | 600 | X1, serial retention, rare core alleles and environmental survival controls. |
| A copying law and its inherited control parameter | 850 | Derive the overlap result; separate payload count from site score; show offset inheritance and prospective law test. |
| Environmental sorting of reproductive designs | 450 | Give the strongest L=32 contrast, updated DZ evidence and causal limits. |
| Different machines, different routes | 500 | 6502 census and honest S1 verdicts; prospective 8086 route result; BFF as the counterexample preventing a universal ladder. |
| Discussion: from reproduction to persistent differences | 300 | Explain biological reach, finite heredity, and the untested step from capacity to adaptive use. |
| **Total** | **3,500** | |

Methods target: approximately 3,000–3,500 words, centered on semantic contracts, assay definitions, provenance, prospective sampling and validation. The main paper gets one explanatory mathematical result, not a procession of numbered elementary observations.

## 9. Extended Data, SI, and material to drop

**Extended Data should answer likely objections without hiding central counterexamples.** Allocate approximately eight displays: (1) validation coverage and I1/I1-T2 component ledger; (2) complete serial/branching and threshold sensitivity; (3) core-allele inheritance and copy-law positional audit; (4) reset dependence, population sampling and modal-marker failures; (5) R/T competition, core loss and rejected ecological mechanisms; (6) validated censuses and 6502 corrections; (7) BFF literal/bandwidth/halting controls; (8) one genealogy synthesis with discovery versus N6 confirmation. Put the 8086 contract sensitivities with the census/validation displays as space allows.

**SI carries the complete scientific record:** proofs and computed certificates; all genotypes and exclusions; full registration chronology and original/updated verdicts; unsuccessful hypotheses; the early length/ablation/padding atlas; genealogy definitions and withdrawn interpretations; raw-data dictionaries; complete manifests, vector coverage, divergence ledgers and performance measurements; reproducible figure commands. Archive negative results rather than writing them out of history.

**Drop entirely from the argument:** universal pattern→self→genome chronology; “three selves coincide”; the noninterference iff; zero hereditary information inferred from zero site score; literal neutral manifolds or bifurcations; parity as a general mixture law; “selection for evolvability”; phase-transition claims without scaling; interaction-as-superior-search claims; the withdrawn recombination/adaptive-walk story; detector benchmarking as a separate attack on biosignatures; unsupported thermodynamic or open-endedness rhetoric. Keep necessary correction records in SI. Drop the separate Zilion-paper plan and product roadmap from the publication strategy.

## 10. Which theory earns its place

1. **Main text: the restricted overlap-copy result.** Give its operation, hypotheses, a short derivation and the exact meaning of `r=d−c`: independently arbitrary payload coordinates **with the core fixed**. State that core alleles may also be inherited. For Z80 retain Proposition 8's initialization, harmless-first-instruction, coverage, alignment and `d|L` conditions. Do not replace them with “any copier with a c-byte core.”
2. **Methods: a copy-provenance certificate.** Unroll a qualifying copying execution into source-coordinate dependencies; identify free coordinates that reach descendants and the invariant family in which the proof repeats. Noninterference is one sufficient route to that certificate. Address-changing variants and non-copy arithmetic fall outside that simple certificate, not outside heredity. Fix eligibility from code/trace before observing mutation retention.
3. **Methods/SI: literal-channel obstruction and construction.** State the corrected repeated-pair bound `N≥2w+2−gcd(2,w)`. Include the proved width-one impossibility for periodic one-register literal writers and the width-three mixture counterexample. The finite-factor width-one witness and its failed zero-partner bootstrap are indispensable scope checks. These results constrain candidate mechanisms; they do not predict population arrival order by theorem. [Independent check](../../results/literal_channel/CHECK.md)
4. **Methods: channel and assay definitions.** Define survival, exact retention, thresholded site count and partner-conditioned output entropy separately. Explain why fixed-parent inflow is not transmitted parental information and why cyclic alignment can change identity claims. The `256^r` family count is a restricted ideal capacity, not an empirical information estimate.
5. **SI only:** BFF copy-cost and confined-address pigeonhole results; finite short-word classifications; motif-window counting; the conditional order framework; the incomplete opportunity/closure model. Full proofs accompany the one main result. Lemmas 10/11 and Proposition 12 are proposals in the narrative notes, not established universal replacements for audited Proposition 8.

**Theory programme for this paper:** finish the copy-provenance certificate, resolve the population-law metric mismatch, and make a prospective prediction on independently implemented 8086 copy geometry. A complete ecological equilibrium theory, adaptation on a functional payload, founder-search theory, error thresholds and critical phenomena are separate programmes unless an inexpensive, parameter-free held-out prediction actually succeeds. Do not add mathematics merely to increase theorem count.

## 11. Zilion is introduced here

Name Zilion in the abstract, give it the second substantive main-text section and Figure 1, and supply its release and evaluation in this paper's Methods/SI. Its contribution is an inspectable, versioned execution substrate for comparing reproductive mechanisms across instruction sets. The strongest demonstration is an independently runnable encounter and heredity benchmark, not a peak-speed number.

The **semantic contract is part of the method**: core/version/variant/compatibility; address-width reduction and ring aliasing; reset registers, flags and stack phase; opcode, operand and prefix fetches; instruction and REP budget units; trap, interrupt, I/O, undefined-opcode and prefetch policies; mutation policy; pairing; writeback; partner distributions; transfer and alignment rules. Distinguish instruction-start maps from all fetched-byte maps. Distinguish externally supplied resources from encoded copying machinery.

Report validation at four levels: instruction semantics, small-world encounter behavior, isolated reproduction/retention assays and ecological outcomes. Passing hardware vectors does not prove a self-modifying soup equivalent to physical hardware. A CPU implementation derived from the same core is not an independent oracle. Cite exact coverage and exclusions.

Historical Algocell kernels, legacy Z80 compatibility and newer C1/Zilion results retain their identities. Pair replay and whole-soup reproduction are different guarantees: collision races prevent claiming deterministic historical soups from a seed alone. The shared packed-word mutation issue remains disclosed as mutation **attempts**, with its population effect unmeasured. A clean mutation policy for new work must be a declared new contract.

The release should include a small deterministic encounter bench, frozen representatives and counterexamples, exact result manifests, and main-figure reproduction scripts. A browser byte-edit/trace demonstration is useful but optional. Do not hold the paper for a general ecosystem, an agent skill, or a new package architecture. Use the [implemented Zilion architecture](/Users/neo/repos/zilion/docs/ARCHITECTURE.md), not the older design roadmap, to describe present capabilities.

## 12. Roles of the 6502 and 8086

### 6502: an honest comparative success with a failed principal component

The validated finite census informed the registration before the soups. It found no nonzero exact zero-partner self-writer at periods 1–4 for L=16 in the declared arms; near-copiers and rare context-dependent open copiers remain possible. Constructed six-byte loops satisfy `d−6` in 296/296 qualifying rows. [Validated census](../../results/census6502/CENSUS.md)

In S1 arm A, all 20 first detected replicators are confined, but only 14/20 satisfy the registered numeric-PC-decrease criterion. **P1 remains not supported.** Ring-position analysis explains the other six after the fact. P2 and slower-arrival P4 are supported; the median detection time is 1,125,000 steps. P3's evolved-offset component is untested: zero eligible measurements, despite observed regenerator/transmitter classes. Do not write “the 6502 inherited payload from birth” without the missing assay. [Correction](../../results/s6502/CORRECTION.md)

The 6502 establishes route contingency and a constructed realization of the copy law. Direct retention on a frozen panel of its evolved first and final copiers is a high-value addition. It is not a substitute for the explicitly prospective third-machine test.

### 8086: prospective route test and prospective law test, scored separately

**Current state matters.** The extant 8086 census is partial and unvalidated: 55,937 unequal-byte words evaluated, 9,343 unknown, constant words skipped, 29 reported writers. Its 13 literal-channel entries and ten one-pass memory copiers are exploratory leads. They must not be called a validated census or used as completed confirmation. The in-progress core targets hardware-recorded 8088 vectors; at this audit its descriptor is still `skeleton` and no completed 8086 baseline ledger is present. [Exploratory research](../ISA_RESEARCH_2026-10-10.md), [independent census audit](../../results/literal_channel/CHECK.md)

Those leads already undermine “open zero-capacity pusher first, then REP core.” Some preliminary one-pass copiers overwrite what they later execute and show zero partner inflow despite open fetches. Zero inflow does not itself prove payload heredity, but it does remove the proposed necessity of a closing transition. Even the preliminary random-register survivor list contains a far-jump word incorrectly called literal. Classify actual traces, not opcode folklore.

**The registration must follow this sequence. This charter is the design, not a falsely completed preregistration.** Final numerical route predictions cannot honestly be written *from a validated census* until that census exists.

**Gate 1 — finish and freeze the machine.** Publish hardware-vector coverage and divergence rules; validate ring encounters, bootstrap, word-write order, PUSH SP, interrupt-vector aliasing, segment and offset wrap, REP zero-count and budgets, and self-modifying execution under the chosen prefetch policy. Hardware 8088 recordings do not certify 8086 bus timing or its prefetch queue. Declare the actual abstraction. In particular:

- The exploratory SP=0 convention writes the first word at `P−2,P−1`; the developing generic “lastByte” convention can put its low byte at `P−1` and high byte at 0. That can change the founder, so pin and retest the phase.
- A test with both tapes already tiled does not establish bootstrap into a zero or random partner.
- `REP MOVS` with CX=0 copies nothing; candidate loops require count setup. State whether REP retains decoded operands across iterations, and count iterations as execution work.
- Fix DS/ES, DF and TF explicitly. Randomizing FLAGS is not a clean pointer-initialization intervention.

**Gate 2 — validated census before any soup.** Enumerate all 256 period-one and all 65,536 period-two words, including constants and both phases, at L=16/32 and budgets 128/512, against zero and frozen random partners. Report primitive-period duplicates transparently. Require no unevaluated words for an exhaustive claim. Classify by traced mechanism: literal writer, memory-fed one-pass copier, confined copy loop, other. Measure write coverage and partner dependence separately. Audit the candidate `B9 51`, `FF 30`, `AD 50` and REP constructions with the same final contract. Archive the counts and hashes.

**Gate 3 — write predictions from those validated results, then freeze.** Before observing any population trajectory, publish candidate classes, mechanism-specific route directions, empirical frequency thresholds, seeds, horizon, sampling schedule, censoring and failure rules. The law eligibility extractor and mutation scorer must also be frozen. No post-soup repair of the route narrative. Throughput benchmarking may use fixed inputs; it must not become an unreported population pilot.

Recommended cohort design is 20 independent worlds at L=16/128 and 20 at L=32/128, with a fixed three-million-step horizon; add 20 matched L=16 worlds with controlled pointer-register initialization if the validated census establishes a differential. L=16 is the primary route cohort; L=32 is a predeclared secondary scope test, scored separately with the same criterion, never pooled to rescue the primary result. Retain the declared lattice/encounter intensity from the comparison protocol and state mutation attempts versus applied updates. Freeze actual manifests and sampling cadence before launch. The numerical sizes here are proposed design choices, not results or rate predictions derived from motif counts.

**Route predictions to finalize at Gate 3:**

| Validated census finding | Prospective population hypothesis | Verdict discipline |
|---|---|---|
| Short, robust open writers remain available under the frozen default reset. | Primary proposed endpoint: at least 80% of first detected copiers are open, with at least 10/20 worlds producing a detected copier by the horizon. Literal and memory-fed subtypes are reported separately. | Availability motivates this empirical threshold; it does not prove 80%. Below ten detections is uninformative about route frequency and informative about arrival. A failed 80% criterion stays failed. |
| One-pass memory copiers survive the validated census. | Register their occurrence and persistence as a separate endpoint; do **not** require a pusher→confined succession. Choose and publish its quantitative threshold from the completed census rationale before soups. | Open, zero-inflow persistence cannot be relabeled a failure to finish the expected transition. Its allele retention is measured independently. |
| Pointer-register randomization selectively removes those candidates while sparing verified literal candidates. | Predict a lower fraction of first copiers using the memory-fed one-pass route in the randomized arm; freeze an effect-size threshold and exact world-level test before launch. Hold flags, segments and stack phase controlled. | If the validated census does not show the differential, withdraw this proposed comparison **before** dynamics; do not run it as if the premise held. |
| The validated candidate inventory differs materially from the exploratory one. | Rewrite the predicted route from the new inventory and explain the change before any population run. | Never retain 29/13/10 as authoritative counts or force the machine into the Z80 story. |

Existence counts alone do not predict establishment or times. This test earns reach by risking a clear, frozen population prediction, not by pretending a census is an arrival theorem.

**The prospective law test is more exact.** Select evolved copiers through a fixed world/snapshot sampling rule that includes random cells, not only high-frequency modal tapes. Report all candidates and exclusions. Determine source provenance, displacement, constrained footprint and eligibility **before** retention assays. Do not infer `c` by subtracting observed sites from `d`.

For the candidate `BF dLo dHi 49 F3 A4` (`MOV DI,d; DEC CX; REP MOVSB`), the proposed six-byte-core prediction requires DS=ES=0, SI=0, DF=0, constant forward displacement, `6≤d≤L`, `d|L`, enough writes, a preserved copied core, and no interfering operation. If verified, predict exactly `d−6` independently free payload coordinates outside the core: for L=16, d=8/16 gives 2/10; for L=32, d=8/16/32 gives 2/10/26. These are prospective conditional mathematical predictions, not assumed evolved genotypes. Other evolved copy geometries receive predictions from the frozen dependency rule or are excluded with reasons, not fitted to `d−c` afterward.

Test every alternative byte value at predicted payload positions, combinations of values, erased-position controls, core/displacement inheritance and serial passage through generation eight with independent fresh partners. Include a fixed constructed REP panel as a positive control, starting from d-periodic parents with the core at the offspring entry phase. Score causal payload transmission separately from whole-tape similarity so that correct regeneration is not mislabeled erasure by the 75% copying gate. For exact qualifying operations, one reproducible, provenance-resolved mismatch in admissible viable descendants requires a theorem/application/implementation diagnosis; exclude partner reacquisition and distinguish a mismatch from stochastic threshold-score disagreement. “80% agreement” is not the exact law's pass criterion. Replicate candidates across worlds to assess how much of the evolved population the law covers.

Keep three outcomes separate: **wrong route distribution = failed route prediction; no eligible evolved core = untested evolved-law prediction; eligible, reproducible retention mismatch = failed law application.** Constructed success cannot silently replace missing evolved evidence. The third machine belongs in the paper whichever outcome occurs.

## 13. Experiments and audits before submission, ranked by claim strengthened per effort

| Rank | Work | Payoff and completion criterion | Relative effort |
|---|---|---|---|
| 1 | Direct offset-inheritance and reciprocal-edit assay on archived evolved Z80 parents | Follow full core haplotypes, actual descendant displacement and new payload retention through eight transfers; include reversion and fresh-partner controls. Converts the compelling marker evidence into the complete causal statement behind B. | Small |
| 2 | Reconcile the copy-law population table | Audit positions and eligibility for the 53 excess-total-site cases and the single deficit; distinguish non-core count, sampled threshold score and reproductive viability. Recheck all alternative values on a frozen representative panel. Stops an avoidable metric error from undermining the exact law. | Small |
| 3 | Serial and joint-payload assay on a frozen panel of evolved 6502 copiers | Recover genuine copy geometry from read/write provenance; predict before assays. Report no eligible architecture if that is the result. Provides cross-ISA evolved retention without another soup campaign. | Small–medium |
| 4 | Complete validated 8086 census and constructed law controls | Mandatory bridge from the already authorized core effort to honest prospective predictions; settle bootstrap, one-pass alternatives and REP semantics. | Medium after core validation |
| 5 | Registered 8086 route and evolved-law cohorts | Largest increase in prospective reach; retain separate verdicts and all exclusions. New architecture can falsify the route while corroborating the mechanism. | Larger; required by the chosen paper design |
| 6 | Neutral tags introduced once into extant transmitter lineages in soup | Track material ancestry, core and tag jointly; eliminate continued partner reacquisition. Establishes ecological use of the storage channel for lineage information, not useful function. | Medium |
| 7 | One held-out environment-switch/offset competition with matched starting haplotypes | Test a predeclared design-frequency direction; record absolute survival and offset alleles. Strengthens environmental sorting while revealing whether payload background explains the original contrast. | Medium |

Ranks 1–2 are submission requirements for the sharpened B claim; ranks 4–5 implement the user's third-machine decision. Rank 3 is unusually good value and should be done. Ranks 6–7 are optional if the existing central chain is clean. A functional payload would answer a further question about adaptation; it is neither necessary to establish capacity nor replaceable by a neutral tag. No broad new ablation atlas, critical-exponent study or universal ecology model is needed before submission.

## 14. Updated evidence and reporting commitments

I1-T2 preserves **19/22 claim verdicts and changes 4/36 components**. “Stand” includes negative and no-direction verdicts. DZ2 changes from not met to met with two reversals reduced to one; DZ3's p=1 majority endpoint changes from 6/10 to 8/10. L2 changes to not applicable at L=16 and to not met at L=20; the Stage L joint kill still fires. The survival/retention contrast and the L=32 lethal transmitter result stand. [T2 report](../../results/tier2/T2.md), [registration outcome](../../REVISION_PREREG.md)

These are changed verdicts **under fresh reruns with C1**, not effects causally attributed to the core correction: the original soups were not run-to-run deterministic. Do not erase original failures because a new run passes. Likewise, keep the 6502 registered PC-loop failure and its explanatory follow-up separate. The new literal result proves a restricted width-one obstruction, not absence of every finite 6502 writer. The five period-two Z80 self-writers include only three literal channels; memory-fed forms must stay distinct.

Every headline number should bind to a dataset, semantic contract and scorer. World-level replication, cell sampling, allele sampling and theoretical quantifiers must remain visible. The reader should be able to tell whether a sentence is a theorem, a registered result, a construction, a post hoc audit or a pending experiment.

## 15. Disagreements

These are the remaining decisions for the user, rather than simulated consensus. “Claude” below refers to the written R2 position; “Astra” is the recommendation in this charter.

| Issue | Claude's argument | Astra's argument and recommendation |
|---|---|---|
| **Universal dependence principle** | Noninterference identifies the necessary and sufficient condition for arbitrary inheritance and supplies a general principle beyond machine code. | Same-trace independence is sufficient under additional premises, not necessary. The offset allele contradicts the ordinary slogan; the dispatcher example refutes the all-value iff. **Reject the necessity; lead with the copying transformation.** |
| **Point versus manifold attractors** | A zero-capacity copier restores a point; a transmitter preserves a manifold whose dimension is heredity capacity. | The environment supplies a kernel, finite tapes have no intrinsic manifold dimension, extinction is not attraction, and “zero-site” parents retain working core alleles. **Use a restricted projection/product-family picture without a universal bifurcation claim.** |
| **“Inherits nothing”** | X1 plus straight-line fetching explains zero hereditary capacity in the early pusher. | X1 establishes loss of the tested pusher16 alleles under its protocol. Rare pusher64 retention and inherited regenerator core variants show that the score is not total hereditary information. **Write the bounded assay result, prominently.** |
| **“Capacity is a heritable byte”** | One copied parameter sets `d−c`, making a concrete contribution to the evolution-of-evolvability debate. Claude explicitly does **not** claim that selection favors evolvability. | The inherited parameter is real and is the best sharpening. Remaining disagreement is wording and scope: capacity depends on the whole contract, and useful adaptive variation has not been measured. **Adopt “an inherited displacement controls payload capacity”; retain Claude's explicit selection caveat.** |
| **The 8086 route** | Predict open zero-capacity pushers first, then REP offset cores. | The unvalidated preliminary census already contains plausible one-pass open alternatives, and the stack/prefetch contract is changing. **Let the validated census determine the registered route prediction; do not require later closure.** |
| **What counts as the climax** | The law and prospective 8086 test; ecology opens the next question. | Substantial agreement. Astra gives inherited control of the law a central panel and uses ecology as established reach with incomplete mechanism. **Keep one climax: what reproduction preserves, and how that rule is inherited.** |
| **Payload function before publication** | Capacity suffices; a neutral tag is the smallest useful follow-up, functional payload later. | Agreement on the submission bar. Astra insists a neutral tag demonstrates transmission/lineage information, not selectable function. **Do not sell that follow-up as adaptation.** |

Two former disagreements are settled by the user: Zilion receives a real introduction and full method in this paper, and the 8086 is included. Astra withdraws the R1 companion-tool-paper recommendation. Claude has already withdrawn coincidence of the three selves, general literal parity and the claim that the theorem prospectively predicted the 6502. Neither side should let those ideas return through figure captions or compressed abstract language.
