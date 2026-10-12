# Astra — independent Round 1

11 October 2026. Written independently, without reading Claude's Round 1. This is a synthesis and recommendation, not a proposed replacement manuscript. Evidence is assessed from the supplied record; the launched I1-T2 study is not treated as completed or scored.

## My recommendation

**Make this a paper about what replication transmits.** The strongest discovery is that successful reproduction and the inheritance of variation come apart, and that a concrete copying mechanism explains the difference. Some copiers restore a pattern; others preserve an arbitrary sequence. Both can reproduce reliably. Only the latter preserve the tested changes outside their copying core. In a recurring family, the displacement of an overlapping copy determines exactly how many such positions survive. Ecology then changes which reproductive design predominates.

My preferred working title is **“The copying mechanism sets the capacity for heredity.”** The bolder alternative is **“Successful replication can erase heritable variation.”** Neither should become a claim that regenerators have no possible evolutionary variation, or that the experiment has demonstrated open-ended evolution.

The current title, “Replication precedes individuality, and heritable variation follows it,” tells a ladder story that the complete evidence does not support as a general account. The 6502 begins differently. BFF separates instruction-pointer confinement from environmental independence. A confined regenerator can be an excellent copier and a poor transmitter of mutations. These are not awkward exceptions to conceal in the supplement. They tell us which concepts the paper must separate.

There is a strong paper here. Its strength is **mechanistic discrimination and experimental explanation**, not spontaneous replication alone, not an unprecedented succession from stack writers to block copiers, and not the depth of the current mathematics. Those first two observations already have close precedents. Most present proofs are elementary. That is acceptable: a short theorem that explains thousands of measurements is more valuable here than an impressive theorem disconnected from them.

Zilion should enter early as the instrument that makes this discrimination possible. The arc can honor the user's vision: we built an experimental substrate; it exposed a consequential ambiguity in what counts as reproduction; a copying law resolves that ambiguity; ecological interventions and a second machine delimit its reach. The technology should earn attention through what it lets us establish.

## 1. Evidence ledger

### Reading the ledger

Paths below are relative to `experiments/`, except explicit absolute paths. `R/` means `results/`, `M/` means `manuscript/`, and `P/` means `planning/`. `REG` means `REVISION_PREREG.md`; early stage registrations are also recorded in `PLAN.md` and `THEORY.md`. Generated reports and correction records take precedence over obsolete prose.

**Status:** **C** = registered and confirmatory; **F** = registered but not met, including a component that could not be tested, explicitly identified; **E** = exploratory, including a post hoc explanation of registered data; **D** = descriptive, including implementation facts, mathematical statements and pending work. Registration of a run does not make every later analysis confirmatory. A failed prediction can yield a strong discovery. Surprise is relative to the prior art and the project's own expectations, not to a reader seeing artificial life for the first time.

**Generality:** **1Z**, **1B**, **1M** = one machine/substrate, Z80, BFF, or MOS 6502 respectively; **2** names the two compared substrates; **T** = theorem under its stated hypotheses, not unrestricted universality. The 8080 opcode subset is a Z80 ablation, not a second independently implemented processor. Descriptive validation counts are coverage claims, not proofs that an emulator is flawless.

### A. Establishing what appeared

| Finding, one claim per row | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| Stages A–M contain 3,660 numbered Z80 runs, but differing endpoints, early stopping and later amendments prevent treating these as one homogeneous confirmatory sample. | D | Low: scale is an asset, not a result. | 1Z | M/MAIN_nature.md, Methods; R/review_r2/RUN_TABLE.md |
| The original occupancy endpoint can fire on zero floods without replicators; Stage B's literal “no-copy means no occupancy crossing” prediction failed in 10 runs although no-copy produced no cultured replicator. | F | High: the first apparent emergence detector detected the wrong phenomenon. | 1Z | R/REPORT_stageB.md; R/detectors/FINDINGS.md |
| Zero-byte material reaches roughly 24–34% mean abundance at the early occupancy event across seven tested lengths. | D | Medium: active restructuring precedes the culture endpoint. | 1Z | R/stageB/NUMBERS.md; M/NUMBERS_INDEX.md |
| The high-order-entropy proxy is near chance or reversed at six of eight tested lengths from 25 to 100 in unablated soups. | E | Medium: a published complexity proxy does not transfer reliably to this assay. | 1Z | R/detectors/NUMBERS_DETECTORS.md |
| A RePair-based assembly proxy exceeds ten times its random reference before the first cultured replicator in 71/80 worlds. | E | Medium: structured products need not imply reproductive lineages; this does not refute every assembly-theoretic biosignature. | 1Z | R/biology/assembly/FINDINGS.md |
| Load–push first replicators occur in all 170 confirmatory benign worlds in the stated L = 16–64 conditions; the result holds at culture thresholds 0.2–0.4. | C | Low for succession itself, medium for its consistency: close precedents exist. | 1Z | R/review_r2/THRESHOLD_GRID.md; M/MAIN_nature.md |
| At thresholds 0.5 and 0.6 the load–push counts fall to 155/169 and 107/164, so “first” depends on what reliability is required. | D | Medium: discovery time and reliable reproduction are different endpoints. | 1Z | R/review_r2/THRESHOLD_GRID.md |
| Confirmatory first-event medians are 450, 700, 425, 600 and 750 steps at L = 16, 20, 32, 50 and 64 respectively. | D | Low: no simple exponential increase with tape length. | 1Z | R/stageG/stageG/NUMBERS_G.md; R/stageK/SCORING.md; M/NUMBERS_INDEX.md |
| Early pushers often make approximate copies and damage themselves; exact copying falls sharply with length despite passing the two-generation culture assay. | D | Medium: the assay supports a reproductive process, not exact cloning. | 1Z | R/stageG/EXACT_COPY.md; R/stageG/NUMBERS.md |
| Removing block copy does not eliminate early replication; the earliest coarse sampling could not resolve the registered narrow timing-equivalence claim. | F | Low: the alternative stack route is known; equivalence was over-specified. | 1Z | R/REPORT_stageA.md; R/REPORT_stageC.md |
| Removing the broad stack/exchange/return arm delays emergence by about 160-fold in Stage C and changes the copying mechanism. | C | Medium: a large intervention identifies an accessible route, but it does not isolate PUSH alone. | 1Z | R/REPORT_stageC.md; R/stageC/FINDINGS.md |
| Targeted ablations show large delays on removing immediate loads or stack machinery; removing other families while retaining load–push does not show a significant timing effect in the small tested samples. | D | Medium: mechanism matters more than instruction-count reduction. | 1Z | R/stageC/stage_c/NUMBERS_C.md; M/NUMBERS_INDEX.md |
| Removing available copying routes produces no cultured replicator within the tested horizons; this is a finite experimental absence, not an impossibility theorem. | D | Low: useful negative control. | 1Z | R/REPORT_stageA.md; R/REPORT_stageB.md; R/stageC/FINDINGS.md |
| At L = 9, suppressing stack-writing machinery improves emergence, reversing its effect at L = 16. | C | High: a locally helpful copying operation can impede another route. | 1Z | R/REPORT_stageD.md; R/stageD/FINDINGS.md |
| A monotone minimum-size story fails: L = 5 and 6 can replicate, L = 8 readily does so, and L = 9–12 includes a difficult region. | F | High: geometry and execution defeat an intuitive complexity floor. | 1Z | R/stageE/FINDINGS.md |
| Holding instructions per byte constant largely flattens emergence timing and restores long-tape copying faithfulness; holding total soup bytes constant does not explain the nominal timing curve. | C | Medium: execution opportunity matters more than the simplest lottery account. | 1Z | R/stageE/FINDINGS.md |
| The broad claim of length-independent emergence time across all control arms is not met, although emergence probability remains high in most large-L conditions. | F | Medium: “flat” depended on the resource normalization. | 1Z | R/stageE/FINDINGS.md |
| At L=100, E4 confirms increased faithfulness with execution budget, but no whole-tape first copier appears in 120 runs, contrary to the proposed high-budget cargo route. | C for faithfulness; F for cargo | Medium: more execution improves a mechanism without necessarily creating a different hereditary design. | 1Z | R/stageE/FINDINGS.md |
| E5 confirms a decreasing maintained period with increasing mutation in the block-copy regime, while the common pusher mostly retains period two over the tested range. | C | Medium: a mutation–structure relationship depends on the copying mechanism; this is not yet a universal error-threshold law. | 1Z | R/stageE/FINDINGS.md |
| At 32 instructions and the specified mutation rate, E7 gives 18/20 cultured replicators with the full ISA and 0/20 after block-copy removal, confirming a budget-specific exception to early dispensability. | C | High: operation availability and execution time interact. | 1Z | R/stageE/FINDINGS.md |
| Tiled first-replicator periods usually divide the pair length in the original geometry, meeting the registered tiling criterion in all eight arm summaries. | C | Medium for the non-pusher arm; low for period-two pushers. | 1Z | R/stageE/FINDINGS.md |
| Padding falsifies the proposed law that emergent periods must divide the memory-ring length; tiled forms also appear on prime rings. | F | High: an attractive gcd explanation was wrong at population level. | 1Z | R/stageF/FINDINGS.md |
| Padding to 28 bytes rescues L = 12 as predicted and L = 10 against prediction; padding L = 8 to 32 bytes reduces emergence to 2/10. | C/F, separate components | High: the same tape length can be easy or hard under different address geometry. | 1Z | R/stageF/FINDINGS.md |
| The registered invariance of load–push emergence to padded ring length fails; odd rings can delay and disrupt its tiling. | F | Medium: even the simple route depends on the world contract. | 1Z | R/stageF/FINDINGS.md |

### B. Confinement, inheritance and competition

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| L = 16 reaches confined majority in 20/20 worlds, L = 20 in 19/20, L = 64 in 8/20; these are condition-specific endpoints. | D | Medium: confinement is frequent but not inevitable. | 1Z | R/closure/NUMBERS_CLOSURE.md; R/stageG/FINDINGS.md |
| L = 32 closes in 12/20 worlds, below K2's required 15; the registered kill threshold below ten is also not reached. | F | Medium: neither a success nor a killed hypothesis by the stated rule. | 1Z | R/stageK/SCORING.md |
| L = 16 converges to the same RET NZ eight-byte motif in 17/20 final dominants. | D | Medium: strong repeatability of a specific accessible solution. | 1Z | R/stageG/ZOO.md; M/NUMBERS_INDEX.md |
| All 89 sampled loop-bearing final dominants are fetch-confined; 85 have zero measured partner inflow, while four retain one to three partner-derived bytes. | D | Medium: confinement and informational independence usually coincide here, but not identically. | 1Z | R/exectrace/AGREEMENT.md; R/biology/individuality/NUMBERS_INDIVIDUALITY.md |
| At L = 50 a jump marker reaches about 0.74 abundance while actual confinement remains about 0.19–0.25 in the invasion check. | F | High: motif takeover was not takeover of the claimed phenotype. | 1Z | R/marker_check/REPORT.md; REG, V |
| The L = 50 final “closed” lineage reopens and fails copying at the fourth transfer, correcting the earlier third-transfer statement. | D, corrected | High: a successful one-encounter genotype is not a stable reproductive lineage. | 1Z | R/serial_retention/REPORT.md; M/RESPONSE_TO_REVIEWS_R2.md |
| Properly classed aligned-length closers seeded at 1% displace pushers in 5/5 worlds at each of L = 16 and 32; reverse pusher invasions fail. | C | Medium: functional advantage is established in these competitions. | 1Z | R/invasion_aligned/REPORT.md; REG, B |
| First-generation mutant scans do not show the predicted general increase in robustness or transmissible positions from first to final dominant. | F | High: reliable copying can remove rather than preserve perturbations. | 1Z | R/mutscan/REPORT.md; REG, R1 |
| In the serial panel, 96–100% of open-pusher lineages die by transfer eight and no position remains transmissible; the prediction of persistent open-pusher transmission fails. | F | High: single-generation apparent heredity does not persist. | 1Z | R/serial_retention/REPORT.md; REG, S |
| Closed regenerators can retain the reproductive process while erasing almost all tested mutant alleles; transmitters retain mutations at 9–20 positions, including unexecuted bytes. | C | High: continued replication and retention of variation are experimentally separable. | 1Z | R/serial_retention/REPORT.md; R/review_r2/LINEAGES.md |
| X1 branching rescues unmutated pusher16 populations: 48/64 survive eight generations in the registered uniform-partner arm. | C | Medium: serial lineage extinction was not enough to establish failure of population persistence. | 1Z | R/branching/BRANCHING.md; REG, X1-1 |
| X1's prediction of at least one transmissible pusher16 site fails: zero of 512 mutant types retain the allele at generation eight across 2,048 populations in each partner arm. | F | High: reproduction can persist while the tested variations disappear. | 1Z | R/branching/SUPP_TABLE.md; REG, X1-2 |
| X1's controls distinguish a transmitter with nine persistent sites from two regenerators with zero, meeting the registered control criterion. | C | High: the contrast survives the branching alternative explanation. | 1Z | R/branching/BRANCHING.md; REG, X1-3 |
| Pusher64 has zero sites by the registered criterion but rare retained mutant types, 5/2,048 and 18/2,048; “no mutation ever survives” is false. | D | Medium: site-level inheritance and rare allele survival are different claims. | 1Z | R/branching/SUPP_TABLE.md; M/NUMBERS_INDEX.md |
| An independent X1 reimplementation supports the qualitative result but estimates lower unmutated survival with more trials; kin-pool survival is inflated by partners already resembling the parent. | E | Medium: the main inference survives, but the kin survival number is unsuitable as headline evidence. | 1Z | REG, Outcome of X1; M/NUMBERS_INDEX.md, discrepancy 12 |
| In 24 surviving parent-by-environment combinations, regenerators retain zero sites and transmitters 9–12; environmental changes mostly determine whether those parents survive. | C | High: the heredity mode follows the copying design across substantial environmental changes. | 1Z | R/pe/PE_MATRIX.md; REG, PE |
| PE's prediction that the pusher loses all serial lineages in every environment fails: about 0.36 survive with closed-soup partners, still with zero transmissible sites. | F | Medium: population context rescues persistence without rescuing the measured inheritance. | 1Z | R/pe/PE_MATRIX.md; REG, PE2 |
| X2 finds 3.78 bits median offspring variability against whole-soup partners at L = 16 and partner fetch in every encounter for all 80 first replicators. | C | Medium: uniform random bytes are not the sole cause of partner dependence. | 1Z | R/inflow_insoup/INFLOW_INSOUP.md; REG, X2 |
| Actual lattice neighbours are kin in about 57% of open-phase encounters, versus 11% under mixing; X2 did not measure information flow under that local neighbour distribution. | E/D | Medium: the laboratory partner distribution remains a substantive limitation. | 1Z | R/stageH/KIN_CENSUS.md; M/NUMBERS_INDEX.md |

### C. Reset conventions, the copying law and ecology

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| The 8080 opcode-subset ablation begins open at both tested lengths; it closes at L = 16 in 20/20 worlds and at L = 32 in 0/20 within the horizon. | C | Medium: instruction availability changes accessibility without establishing impossibility. | 1Z | R/stageM/SCORING.md |
| A constructed 20-byte-core closer works under that subset and preserves all 12 payload positions over eight transfers, refuting the proposed impossibility of an L = 32 closer. | F | High: the earlier nonexistence explanation was constructively false. | 1Z | R/invasion_closer/A1_REPORT.md; R/serial_retention/REPORT.md; REG, A1 |
| A1's exact-motif invasion endpoint fails in all five worlds; a post hoc functional analysis shows confined descendants prevailing, while the full-Z80 regenerator defeats the constructed closer in A2. | F for A1; E for descendants; C for A2 | Medium: genotype markers can lose identity while their reproductive phenotype spreads. | 1Z | R/invasion_closer/A1_REPORT.md; R/invasion_closer/A2_REPORT.md |
| Random initial registers preserve the open beginning in 20/20 worlds but delay confinement strongly; pooled longer runs yield 8/40 closures by three million steps. | D | High: the harness supplies useful initial conditions to common evolved closers. | 1Z | R/r4/R4_LONG.md; R/r4/R4_REP.md |
| C4a's “at most five” long-run closures fails at 6/20; its replication yields 2/20 within the registered 2–12 interval, and the apparent odd-seed effect disappears. | F then C | Medium: a modest count and a seed pattern must not become a mechanism. | 1Z | R/r4/R4_LONG.md; R/r4/R4_REP.md; REG, C4 |
| Random stack initialization still produces 18/20 open first replicators; 13 use the operand-free stack mover `e1 e3`, so a literal channel is not necessary for an open beginning. | C/D | High: another route defeats the narrow literal-channel dichotomy. | 1Z | REG, C3; M/MAIN_nature.md |
| Only the constructed self-initializing closer survives the random-register environments in the tested parent panel. | C | Medium: self-initialization can replace the scaffold, but the common designs do not do so. | 1Z | R/pe/PE_MATRIX.md |
| Under random registers, C4b's exact self-initializer motif-majority prediction fails 0/5, while confinement and reverse-pusher suppression meet their 5/5 criteria. | F/C, separate components | Medium: functional success and a frozen sequence label diverge again. | 1Z | R/r4/R4_INVASIONS.md; REG, C4b and I1-T2 |
| Stage L closes in all 20 worlds, but the predicted recovery of the main modal genotype's transmissible positions fails and the corresponding kill criterion fires. | F | High: long continuation after closure does not force the winning genotype to preserve a payload. | 1Z | R/stageL/SCORING.md |
| Stage L's registered unexecuted-site rule for recovering lineages and its broad exclusion of variation in return/push dominants also fail; later restriction to closed non-block forms does not rescue every condition. | F | Medium: coarse mechanism labels and one-generation sites do not support an absolute exclusion. | 1Z | R/stageL/SCORING.md; REG, I1-T2 component rules |
| Population scans find transmitters missed by exact-modal-tape sampling; some transient transmitters occupy less than 0.3% of the soup. | E | High: the detector preferentially sees genetically uniform regenerators. | 1Z | R/population/REPORT_classes.md; R/stageL/SCORING.md |
| For the specified four-byte Z80 core, displacement d dividing L yields a tiling of the first d bytes and exactly d−4 universally transmissible non-core positions under the proposition's assumptions. | D, mathematical | High explanatory value, low mathematical difficulty: one operation gives the mechanism. | T | M/MAIN_nature.md, Proposition 8 |
| The law matches 2,289/2,343 eligible sampled cells and all 15 group medians; these are correlated cells, not 2,343 independent evolutionary trials. | E | High: a small local law explains a broad empirical pattern. | 1Z | R/offset/COPY_PERIOD_LAW.md |
| The recurring `XX 5e ed b0` family is not the commonest confined family overall: 2,347/15,655 sampled confined heritable cells carry it. | D, corrected | Low: scope correction essential to the generality claim. | 1Z | R/offset/CORE_FAMILIES.md; M/RESPONSE_TO_REVIEWS_R2.md |
| Changing displacement can switch regeneration to transmission; the ecological R/T seed comparisons also differ in payload and are not pure one-byte matched-genome experiments. | D | High mechanistically; the causal intervention must be described exactly. | 1Z | R/offset/REPORT.md; M/MAIN_nature.md, Proposition 8 |
| In a matched L = 32 comparison, benign worlds contain roughly 0.03–0.37 transmitters and lethal worlds 0.55–0.87; the lethal mixture/convergence predictions are met. | C | High: the environment changes the prevalence of a capacity for heredity without rewarding a payload task. | 1Z | R/offset/SWITCH_TIMES.md; REG, E-L32L |
| The initial E1/E2 quantitative equilibrium predictions fail; turning mutation off yields approximately neutral R/T competition rather than the predicted deterministic equilibrium. | F | High: a copying law does not by itself determine ecological selection. | 1Z | R/offset/REPORT.md; R/offset/OFFSET_ENCOUNTER.md; REG, E |
| Core loss occurred in seven, not three, L = 16 lethal worlds; benign L = 16 worlds all fall below the core-count threshold at some time and only seven recover by the reported endpoint. | D, corrected | Medium: conditioning on surviving cores can hide the main outcome. | 1Z | R/offset/SWITCH_TIMES.md; M/RESPONSE_TO_REVIEWS_R2.md |
| Isolated mutational-flow estimates fail to predict the ecological equilibrium: EM1 gives about 0.42 rather than below 0.18; EM2 gives almost identical benign and lethal equilibria. | F | High: differential mutation alone does not account for the rule-dependent mixtures. | 1Z | R/conv/REPORT.md; REG, EM |
| Evolved-soup-conditioned flow estimates recover the qualitative mixture after the fact, but use the environment that the theory should predict. | E | Low as independent explanation: circular closure remains. | 1Z | R/conv/REPORT.md |
| The proposed selected toxic payload is not supported: HALT enrichment and the required detoxification effect fail; one population-conditional prediction was not run. | F | Medium: unexecuted sequence is not automatically a functional toxin. | 1Z | R/toxin/CAUSAL.md; R/toxin/COMPOSITION.md; REG, N1 |
| Backup-core protection fails: zeroing the first core prevents copying in 0/4,096 successes for both tested designs; repeated cores are not working backups in that assay. | F | High: an intuitive redundancy explanation is mechanically wrong. | 1Z | R/n2/BACKUP.md; REG, N2 |
| N4's inherited-zero shield explanation fails: detoxified tails retain about 88–94% of the protection. | F | High: the particular scar is not the protective cause originally proposed. | 1Z | R/n2/SCARS.md; REG, N4 |
| Encounter demography shows nearly balanced direct R↔T conversion and different responses to core-free intruders; measured terms explain much of the drift contrast but not the benign equilibrium. | E | High: the relevant selection acts through third-party interactions, but the account is incomplete. | 1Z | R/n2/DEMOGRAPHY.md; R/n2/MUTUAL.md; R/n2/BACKGROUND.md |
| The registered attempt to reproduce the prior study's greater block-copier mutation robustness fails on these Stage K tapes; a constructed transmitter is robust under that exact-copy assay. | F/E | High: robustness depends on what is copied and what the assay calls success. | 1Z | R/review_r2/ROBUST_REF22.md; REG, F |
| Stage I's predicted open-then-extinct route occurs in 0/10 worlds, while all ten reach closed dominance, firing the registered I3 kill criterion; all first detected copiers use block-copy loops. | F | High: the same processor can begin confined when the stopping rule changes. | 1Z | R/stageI/FINDINGS.md; R/stageI/NUMBERS_I.md |
| At L=16, pooled benign worlds close earlier than lethal worlds by the reported survival analysis, but changing lethality changes more than precursor exposure. | E | Medium: supports an association, not a clean causal effect of an open phase. | 1Z | R/review_r2/FIRST_CLOSURE.md; M/NUMBERS_INDEX.md |
| DZ confirms a decline of open first replicators with tar lethality: 10/10 at p≤0.1, 2/10 at 0.3, and 0/9 detected replicators at p=1. | C | High: even with the same ISA, the early route can disappear. | 1Z | R/dial/DIAL.md; REG, DZ1 |
| DZ's monotone delay-to-closure prediction fails with two median reversals, despite a positive aggregate trend. | F | High: integrated exposure alone is not yet a predictive window theory. | 1Z | R/dial/DIAL.md; REG, DZ2 |
| DZ confirms increasing transmitter prevalence with lethality, but the p=1 majority endpoint is 6/10 rather than the required seven. | C for trend; F for endpoint | High: ecological influence on heredity is real, quantitative prediction imperfect. | 1Z | R/dial/DIAL.md; REG, DZ3 |
| The top-three, 0.5%-share closure endpoint misses diverse confined transmitter populations in two DZ worlds. | D | Medium: apparent absence can be detection bias. | 1Z | R/dial/DIAL.md; M/NUMBERS_INDEX.md |

### D. BFF as a mechanism intervention

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| Standard and wrapping BFF have 9/24 and 19/24 detected transitions; all 28 first replicators bear loops, but only 23/28 meet the ≤5% pointer-entry criterion. | C for loop mechanism; F for strict universal confinement | Medium: loop-bearing and pointer-confined must not be conflated. | 1B | R/bff/FINDINGS.md; R/bff/NUMBERS_BFF.md |
| Adding the engineered two-byte literal writer produces loop-free open first replicators in all 12 wrapping and all 12 non-wrapping worlds. | C | High: a surgical language intervention changes the observed route into replication. | 2, Z80/BFF | R/bff/FINDINGS.md; REG, BFF variants |
| With lethal unmatched brackets, the literal replicator collapses; with harmless brackets and wrapping it persists at about 98% through the horizon, with no observed closing transition. | C | Medium: halting ecology controls persistence, and long exposure does not guarantee closure. | 1B | R/bff/FINDINGS.md; R/bff_dials/FINDINGS.md |
| Harmless-bracket BFF without the literal gives seven transitions rather than the predicted nine of twelve, and one first replicator is fully pointer-open while loop-bearing. | F | High: absence of a literal channel does not force pointer confinement. | 1B | R/bff_stdnh/FINDINGS.md; REG, B5 |
| Four of those seven harmless-bracket populations lose detectable replication again by the endpoint. | E | Medium: appearance and establishment are distinct. | 1B | R/bff_stdnh/FINDINGS.md |
| D1's simple inverse-lethality collapse expectation fails: p≤0.1 persists, and at p=0.3 all six collapse transiently but five recover above 10%. | F | High: a threshold/recovery response replaces a simple exposure law. | 1B | R/bff_dials/SCORING.md; REG, D1 |
| Increasing literal bandwidth to r=2 or r=3 gives zero measured partner inflow while the instruction pointer still enters the partner in every encounter. | C | High: execution confinement is unnecessary for this form of informational independence. | 1B | R/bff_dials/SCORING.md; REG, D2 |
| Searches find no qualifying closed copier in the restricted `{P,[,],nop}` grammar through length ten and no period≥2 copier in 16,447,860 searched straight-line BFF candidates. | D | Medium: useful finite exclusions, not a proof about all BFF programs. | 1B | R/bff_closed_search/NUMBERS_CLOSED_SEARCH_p10.md; R/bff_search/NUMBERS_SEARCH.md |

### E. How confined founders were made

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| The original N3 copying criterion counted damage between similar tapes as copying; its ancestry prediction was met but its apparent founders often did not reproduce. | F/D, corrected | High: provenance is only as meaningful as the phenotype definition. | 1Z | R/lod/registered_rule/LOD.md; R/lod/amended_rule/LOD.md; REG, N3 |
| A corrected functional pipeline identifies 17 closed worlds in the initial twenty; analysis choices for this set were made after inspection. | E | Medium: discovery set, not retrospective confirmation. | 1Z | R/lod/FOUNDERS2.md; REG, N3 amendments |
| Recorded encounters replay byte-for-byte: the initial validation checks 795,934 pairs, and the later audit reports zero replay mismatches across both sets of twenty worlds. | D | Medium: unusually inspectable causal histories, not proof of a unique biological parent. | 1Z | REG, N3 validation and N6; M/NUMBERS_INDEX.md, discrepancy 4 |
| X3 finds the earliest copier on every sampled founder line open in 17/17 discovery worlds and 20/20 new confirmatory worlds; lines within a world share ancestry. | E then C | Medium: genuine line-level support, not just successive dominant types. | 1Z | R/lod/OPEN_FIRST.md; R/lod_n6/OPEN_FIRST.md |
| Sparse confined cells can occur away from those lines, so the genealogy result does not identify the first confined tape anywhere in the world. | E | Medium: ancestry-sampled “first” has a precise scope. | 1Z | R/lod/OPEN_FIRST.md; R/lod_n6/OPEN_FIRST.md |
| No point mutation completes the sampled first confined founder in either set, 0/37; the new set has 18 rewrites/copies of the partner and two partial overwrites. | C for N6; E for initial set | High: the birth event is an encounter transformation rather than a one-byte final step. | 1Z | R/lod/FOUNDERS2.md; R/lod_n6/N6_SCORE.md |
| Between the newest open copier and the founder lie only whole-tape non-copiers in 16/17 initial and 20/20 new worlds; median delays are 79 and 46.5 steps. | E then C | High: simple descent through continuously copying intermediates misses the observed route. | 1Z | R/lod/FOUNDERS2.md; R/lod_n6/FOUNDERS2.md; R/lod_n6/N6_SCORE.md |
| N6 confirms enrichment of the founder's executed words: median 0.00190, 7.9 times the stated random baseline; return-family motifs occur in 34/37 pooled founders. | C for enrichment; D for pooled motif | Medium: the open population supplies a biased material distribution. | 1Z | R/lod_n6/PARTS_FIRST.md; R/lod/PARTS_FIRST.md; REG, N6-3 |
| N6-4 fails its executed-code non-copier criterion: 11/20 rather than at least twelve; pooled, 14/37 writers already copy the code they execute. | F | High: “non-replicators make replicators” depends on the unit being copied. | 1Z | R/lod_n6/FOUNDING_SUMMARY.md; R/lod_n6/N6_SCORE.md |
| Whole-tape non-copiers are already about 84–91% of the relevant soup, so their abundance among writers is not significant enrichment under the reported nulls. | E | Medium: a visually striking fact partly follows the base rate. | 1Z | R/lod/FOUNDING.md; R/lod_n6/FOUNDING_SUMMARY.md |
| Recorded point mutations are expected to yield only 0.002 and 0.0064 confined founders under the local reach calculation; N6's <0.05 criterion is met. | E then C | Medium: strong explanation of the recorded paths, not a comparison with all possible mutation searches. | 1Z | R/lod/REACH.md; R/lod_n6/REACH.md |
| Point mutation does create some founder-producing writers, and some writers already copy their executed core; “mutation cannot originate these replicators” is not supported. | D | High: the event that completes a tape is not the entire origin process. | 1Z | R/lod/FOUNDING.md; R/lod_n6/FOUNDING_SUMMARY.md |
| “Recombination,” the event-kind P value and the strong multi-parent assembly narrative were withdrawn after audits; a rewrite label does not imply splicing. | D, corrected | High: causal interpretation changed while the underlying records remained useful. | 1Z | REG, N3 amendments; M/RESPONSE_TO_REVIEWS_R2.md |
| N5's apparent gradual adaptive walk fails stringent copying and two-generation tests, and its invasion is uninformative because the tracked classes disappear. | F/E, withdrawn interpretation | High: approximate similarity created an illusory path to stable replication. | 1Z | R/n5/PATH.md; REG, N5 |

### F. The genuinely second processor

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| The 6502 exhaustive census finds no nonzero exact straight-line self-writer into the zero partner at periods 1–4 for L=16 under the tested budgets and BRK rules; shorter-period checks also cover L=20 and 32. | D, pre-soup census | High: a common short Z80 route is absent in a finite, exhaustive second-machine comparison. | 2, Z80/6502 | R/census6502/CENSUS.md |
| Approximate 6502 writers reach 14/16 bytes and some open programs copy rare random partners; the exact-zero-partner absence does not mean no open copying is possible. | D | Medium: near-copies and context-dependent success remain accessible. | 1M | R/census6502/CENSUS.md |
| Constructed six-byte 6502 copying loops satisfy the displacement-minus-six prediction in all 296 qualifying census rows. | D | High: the overlap mechanism transfers to a second ISA in controlled constructions. | 2, Z80/6502 | R/census6502/CENSUS.md |
| S1 arm A detects twenty first replicators, all fetch-confined, but only fourteen with a non-sequential 16-bit PC decrease; the registered ≥80% conjunction P1 fails. | F | High: the main qualitative contrast is strong but the registered loop endpoint was not met. | 1M | R/s6502/S1.md; R/s6502/CORRECTION.md |
| S1 P2 finds no detected open straight-line precursor and P4's slower-arrival prediction is met, with median 1,125,000 steps versus the Z80 reference 450. | C | High: a prospective substrate comparison changes the route and the observed time scale. | 2, Z80/6502 | R/s6502/CORRECTION.md; REG, S1 |
| All six P1 exceptions return to earlier ring positions by forward branches to higher address aliases; thus 20/20 loop in position space in the exploratory follow-up. | E | High: numeric PC descent was the wrong invariant, but this cannot retroactively turn P1 into a pass. | 1M | R/s6502/EXPLORATORY.md |
| S1 P3 is not tested as a conjunction: final scans contain 94 transmitter and 209 regenerator classifications, but the offset assay has zero eligible cells and measures alignment rather than mechanistic displacement. | F, not testable | High: the observed class mixture does not validate the predicted law in evolved 6502 genomes. | 1M | R/s6502/CORRECTION.md; R/s6502/EXPLORATORY.md |
| Descriptive arms B/C/D detect 4/20, 20/20 and 18/20 first replicators, of which 4, 19 and 18 are confined; arm C has one open loop-bearing case. | D | Medium: the mechanism is robust across several settings but not absolutely uniform. | 1M | R/s6502/v2/S1_v2.md; R/s6502/CORRECTION.md |
| Correcting multi-byte sequential PC wrap leaves all eighty world-level verdict rows unchanged; repeating final classifications at the literal 64 rather than frozen 32 partners also leaves classes unchanged. | D, corrected | Medium: correction tested quantitatively rather than dismissed. | 1M | R/s6502/AUDIT.md; R/s6502/CORRECTION.md |
| Tracing against partners from the nearest saved soup snapshot changes none of the confined/open/loop flags for the 62 detected first tapes; it does not re-test P2's history or the exact contemporary neighbourhood. | E | Medium: useful sensitivity with a clearly limited temporal scope. | 1M | R/s6502/CORRECTION.md |
| The audit exposes a shared packed-word mutation lost-update race; 512 mutations are attempts, and no rerun under a corrected mutation policy establishes its ecological effect. | D | Medium: a common world-kernel property remains unquantified, not a uniquely 6502 defect. | 2, Z80/6502 | R/s6502/AUDIT.md; REG, correction to S1 |

### G. Instrument validation and unresolved impact

| Finding | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| The historical ring differential test reports zero unexplained discrepancies over two seeded sets of 109,000 cases at each of six ring sizes, after documented fixes. | D | Low: valuable coverage with finite reference scope. | 1Z | R/emulator/RING_DIFFTEST.md |
| Zilion's newer Z80 variants have extensive opcode-vector and targeted tests, including 1,604,000 SingleStepTests vectors; compatibility and Zilog-Q semantics are separately versioned. | D | Medium methodological value: semantics are explicit experimental choices. | 1Z | /Users/neo/repos/zilion/docs/ARCHITECTURE.md |
| The 6502 py65 differential study has 2,009,068 trials, 241,797 comparable cases, 233,265 exact matches and 8,532 explained differences, with zero unexplained comparable failures. | D | Medium: an independent implementation supplies an important second check. | 1M | R/py65_difftest/DIFFTEST.md |
| That py65 study excludes 1,767,271 unsupported paths and does not independently validate the 93 non-JAM undocumented opcodes; separate full-opcode vectors and exhaustive decimal tests add different coverage. | D | Low: the denominator limits what “validated” means. | 1M | R/py65_difftest/DIFFTEST.md; /Users/neo/repos/zilion/docs/ARCHITECTURE.md |
| I1 completes 787 conditions ×100,000 pairs per core; 47/690 C1 and 114/690 C2 decision conditions are flagged, so the all-unaffected criterion fails. | F | High methodological consequence: small instruction changes have measurable effects in relevant distributions. | 1Z | R/impact/README.md; REG, I1 |
| I1 finds 407/417 frozen assay comparisons identical and ten different; reported differences are confined to particular B-side copy/preservation columns, not evidence that whole-soup outcomes are unchanged. | D | Medium: reassuring local stability cannot establish long-term ecological robustness. | 1Z | R/impact/README.md |
| I1-T2 has a registered 36-component verdict comparison and a launch of 242 soup worlds plus 21 panel jobs; the supplied launch record contains no final ecological verdicts. | D, pending | High importance, no result yet: central robustness questions remain open. | 1Z | R/tier2/REGISTRATION_DRAFT.md; R/tier2/LAUNCH.md; REG, I1-T2 |
| Zilion separates instruction cores, world semantics and GPU execution, with hashes and explicit budgets/reset/address policies; the soup's ecological scheduler belongs to a higher layer. | D | Medium methodological novelty: the separations make comparative experiments interpretable. | 2, Z80/6502 | /Users/neo/repos/zilion/README.md; /Users/neo/repos/zilion/docs/ARCHITECTURE.md; P/ZILION_DESIGN_2026-10-10.md |
| Pair execution is deterministic under a fixed contract; the historical GPU Z80 soup is not exactly replayable from seed alone, whereas tested BFF CPU/GPU-location comparisons agree bitwise. | D | Medium: reproducibility must specify the level at which it holds. | 2, Z80/BFF | R/zsoup/nondeterminism.md; R/bff/FINDINGS.md; /Users/neo/repos/zilion/docs/ARCHITECTURE.md |

### H. What the mathematics currently establishes

| Result | Status | Surprise and reason | Generality | Source |
|---|---|---|---|---|
| Theorem 1 supplies a per-encounter BFF write/movement cost bound; its earlier claims about unavoidable partner dependence and impossibility after self-modification were withdrawn. | D | Low mathematically; useful accounting constraint. | T | M/MAIN_nature.md, Theorem 1; M/RESPONSE_TO_REVIEWS_R2.md |
| Theorem 2 forces a repeated executed address when a confined run exceeds the available addresses under its no-early-halt/budget conditions; it does not prove recurrence of the entire machine state. | D | Low: pigeonhole principle, not a theory of individuality. | T | M/MAIN_nature.md, Theorem 2 |
| Proposition 3's exhaustive short-word classification is exact for its finite universe; the five period-two Z80 writers include indirect-memory-load forms as well as immediate-load forms. | D | Medium empirical surprise, modest mathematical depth. | 1Z | M/MAIN_nature.md, Proposition 3; R/review_r2/CENSUS_P2.md |
| Proposition 4 counts motif windows but does not establish an exponential arrival advantage: the functional pusher unit and short closer can both require four bytes. | D, corrected | High as a correction; low as a theorem. | T | M/MAIN_nature.md, Proposition 4; M/RESPONSE_TO_REVIEWS_R2.md |
| Model 5's integrated-opportunity formula is a conditional hazard model, not a derived mechanism or a successful monotone predictor across the lethality dial. | D model; F for DZ application | Medium: its value depends on independently predicting production and establishment rates. | 1Z, model | M/MAIN_nature.md, Model 5; R/dial/DIAL.md |
| Theorem 6 gives a conditional construction/exclusion framework; its BFF one-pass exclusion premise remains unproved, and existence of an open copier does not determine first arrival. | D | Low-to-medium: useful hypotheses, excessive if sold as universal chronology. | T, conditional | M/MAIN_nature.md, Theorem 6 |
| Lemma 7 bounds partner-conditioned output variability/fidelity; at fixed parent deterministic execution gives I(E;Y\|X=x)=H(Y\|X=x), not a measure of transmitted parent variation. | D | Low mathematically, high conceptual importance of the distinction. | T | M/MAIN_nature.md, Lemma 7 |
| Proposition 8 is the strongest present explanatory theorem because its exact overlap law connects a copying operation to the mutation-retention data. | D | High reach within its family, elementary proof. | T | M/MAIN_nature.md, Proposition 8; R/offset/COPY_PERIOD_LAW.md |
| Proposition 9's stated Fine–Wilf window is one byte too short for odd widths; a corrected bound can support the restricted repeated-pair argument but not the current unrestricted prose. | D, audit finding in this synthesis | Medium: a repairable proof gap with an important scope limit. | T after repair | P/LITERAL_CHANNEL.md; analysis below |
| Conjecture 9′ about arbitrary mixtures of one-byte literal loads and pushes is unproved; exhaustive periods≤4 do not prove it for arbitrary programs. | D, conjecture | Medium: promising classification question, not settled cross-ISA theory. | 1M finite evidence; proposed T | P/LITERAL_CHANNEL.md; R/census6502/CENSUS.md |

### Consequences for the manuscript's language

1. **Retire the old one-pager, not merely soften its adjectives.** The “inevitably open,” two-byte-versus-four-byte exponential head start, one-jump origin of individuality and theorem-forced sequence in `ONE_PAGER.md`/the earlier plans are superseded explanations.
2. **Say “fetch confinement” where that is what was measured.** The organism-sized tape, encounter boundary, initialization and reset are supplied. No membrane, metabolism or autonomous maintenance of the execution machinery evolved here.
3. **Say “no externally assigned task reward,” not “nothing to select for.”** Overwriting, survival, execution opportunity and encounter scheduling already create selection.
4. **Do not turn an eight-bit sample ceiling into a biological fraction.** Entropy over 256 observed partners cannot justify “seven of the child's eight bits belong to the neighbour.” A 16-byte child is not an eight-bit object, and the parent was fixed in that measurement.
5. **Keep occurrence, establishment and inheritance separate.** A class marker, an exact modal tape, a cultured copier, a persisting lineage and a transmitting lineage are different observables. Much of the project's intellectual progress came from learning this.
6. **Do not advertise universal robustness before I1-T2 is scored.** I1 refuted all-unaffected execution. It neither invalidated all biological conclusions nor established their persistence under corrected semantics.

## 2. The explanatory core

### Thesis A: the mechanism of reproduction determines which differences can survive reproduction

**Short thesis:** A reproductive process can regenerate its own form while erasing changes to that form. Persistent inheritance requires that the process preserve distinctions between parents, not merely produce another successful copier.

This explains the serial/branching contrast, the robust regenerator that rejects its mutant allele, the transmitter's unexecuted payload, much of the parent-by-environment matrix, the misleading modal-genotype readout, and the overlap law. It also explains why “more exact offspring,” “less partner dependence” and “greater evolvability” need not move together. The latter has not actually been measured as sustained adaptive improvement here.

The clean mathematical object is an encounter operator or kernel, `K_e(y | x)`, with an explicit environment/reset distribution `e`. On the permitted family of inputs, a regenerator acts like a projection onto a restricted set of patterns. A payload transmitter preserves many more distinguishable input sequences. Proposition 8 gives an unusually concrete example: copying from zero to displacement d, with overlap and sufficient execution, propagates the first d bytes. When d divides L, the result repeats those bytes. With the four-byte core held fixed, only d−4 positions are universally free to carry arbitrary values. A very small displacement makes an excellent pattern restorer; a displacement of L copies the whole tape.

This is stronger than naming two kinds of organism. It predicts what an intervention at each location does. It is weaker than asserting that the entire genome is partitioned into eternally fixed code and data: the core itself can mutate, instruction/data roles can change, and another environment may execute today's unexecuted bytes.

**Why this is hard to vary.** The explanation is anchored to instruction traces, a precisely specified memory transformation, mutation interventions and multi-generation retention. Replacing “overlap” with “complexity,” “individuality” or “more computation” would lose the quantitative d−4 prediction. The 6502 construction suggests the c-byte-core generalization d−c, but evolved 6502 validation remains outstanding.

**What would break it.** A qualifying core repeatedly preserving arbitrary alterations outside the first d bytes would falsify the exact claim. A mismatch after confirming displacement, initialization and the first opcode's harmlessness would be a real counterexample, not noise. If apparent payload retention were entirely partner reacquisition, lineage-independent background or choice of cyclic alignment, the empirical bridge would fail; the branching controls help but do not replace every in-soup causal test. At the broad level, a single process cannot erase a distinction and also transmit that same distinction through the same channel; that logical statement is secure but uninteresting unless the channel and biologically relevant distinctions are identified independently.

**Predictions beyond the present tests.** Changing overlap while fixing the executable core and an admissible payload should change retention in the predicted locations before any population selection occurs. Restoring an erased variant repeatedly from neighbours should create apparent population memory without lineage memory. A transmitter payload can remain heritable even when it supplies no present reproductive advantage. A perturbation that makes payload bytes execute should convert some retained neutral changes into lethal or regenerative changes. None of these predictions requires a Z80 specifically.

**Reach and new questions.** This connects faithful regeneration to the evolvability problem of autocatalytic systems, distinguishes repair from inheritance, and gives a direct question for chemistry: does a cycle return to a preferred composition, or preserve experimentally imposed differences in composition or sequence? It asks how many *independently perturbed* degrees of freedom survive repeated reproduction, at what rate and under what ecological support. The test is operational; it does not assume genes in advance.

**The incomplete part.** This explains what R and T do, not why their ecological abundances take particular values. The matched halting-rule comparison shows that heredity capacity can be an ecological outcome. The failed toxin, backup and mutation-flow accounts mean we do not yet possess a complete explanation of that outcome. “Selection favors evolvability” would be premature: selection may favor another property correlated with carrying arbitrary sequence.

### Thesis B: the route into replication is set by available transformations and their ecological opportunity

**Short thesis:** Replicators appear through the transformations a substrate makes accessible, then establish only if its execution ecology allows them to persist. Computational universality does not make those routes equivalent.

The literal intervention in BFF, the finite 6502 census and prospective S1 comparison, the lethal-Z80 exception, the random-stack-pointer alternative, the geometry rescues and the genealogies all belong here. This is a more defensible unification than “life first borrows its neighbour's computation, then becomes an individual.”

There are three distinct quantities: the supply of candidate copiers, their probability of establishing, and the variation their reproduction preserves. Counting short programs addresses only part of the first. A literal channel supplies a compact pattern writer; it does not guarantee sufficient budget, harmless partner execution, dominance or payload inheritance. An encounter can create a viable copier from a non-copying tape; it does not follow that encounters outperform mutation as a search process. The 6502 result is valuable precisely because a preregistered comparison survived a change in processor without reproducing the Z80 chronology.

**Why this is hard to vary.** In its current broad wording it is less hard to vary than Thesis A: “accessibility plus ecology” can become a catch-all. Its defensible content comes from named operations and interventions. The BFF literal addition changes a specific capability; ring padding changes address geometry; a halting rule changes what happens when foreign code is run; register randomization removes a concrete scaffold. A successful general theory must predict rates from these changes, rather than assign a fresh rate after observing every outcome.

**What would break the useful version.** A large prospective set of minimal substrates whose route classes do not track the specified operation/budget constraints would defeat the classification. Finding common short 6502 exact straight-line writers outside the finite census would defeat extrapolation, though not invalidate the census. Showing that the same available motifs have equal production and establishment rates while the predicted sequence reverses would reject a particular accessibility account. The existing lethal-Z80 and harmless-bracket BFF cases already break simplistic literal-present/open-first and literal-absent/confined-first laws.

**Predictions beyond the tests.** A literal-width change in a deliberately matched machine should alter the existence of repeated-pair self-writers; changes in write coverage and stopping should independently alter their establishment. A novel machine can contain the same abstract copying operator and show the same retention law despite unrelated opcodes. Seeding equal abundances of candidate-producing tapes may change founder rates without changing isolated copier fitness. These are stronger tests than adding another historical CPU and looking for something familiar.

**Reach and new questions.** Chemical transformations likewise have different accessible products and different context-dependent survival. The appropriate analogy is a reaction grammar plus an execution environment, not “molecules are opcodes.” The open question is which coarse properties survive a change of representation: locality of copying, read-before-write dependencies, dependence on external context, resource-limited completion and retention of perturbations.

### Rivals and what the evidence actually decides

| Rival account | What the evidence decides | What remains open |
|---|---|---|
| The early forms merely fail because one-child transfer kills weak lineages. | X1 rescues pusher16 population survival but not its tested alleles; extinction alone is insufficient. | Larger branching factors, longer horizons, naturally assortative partners and alternative inherited phenotypes. |
| Any confined copier is an evolving individual. | Regenerators erase tested variants; BFF can be pointer-open with zero inflow or pointer-confined with inflow. | A richer causal definition of individuality, including resetting and ecology. |
| Offspring variability is inherited genetic information. | X2 holds the parent fixed; it measures variation caused by partners, not distinctions transmitted between parents. | The full two-input reproductive channel and local-neighbour-conditioned causal information. |
| Closers win simply because they are mutation-robust. | The direct robustness prediction and its literature-matched assay fail for common regenerated tapes. | Different robustness notions, core-mutant changes, frequency-dependent damage resistance. |
| Shorter descriptions inevitably appear first. | Corrected functional lengths undermine the two-versus-four-byte argument; finite censuses contain many closed writers. | A quantified production-rate model using actual execution and encounter distributions. |
| Payload zeros, spare cores or isolated mutation rates explain R/T mixtures. | Registered tests reject the stated versions of all three. | A jointly measured encounter/mutation/spatial model, with held-out equilibrium predictions. |
| Closure is assembled by beneficial point mutations of copying ancestors. | Sampled first founders are completed by encounters, and the proposed N5 adaptive walk fails. | Mutations make parts and producers; unobserved or alternative paths may exist. |
| Program interactions are a uniquely powerful recombination/search operator. | Our records identify actual events, not matched search efficiency; “recombination” was withdrawn. | A resource-matched mutation-only versus interaction comparison. |
| Everything is a bug in one emulator. | Independent tests, BFF interventions and 6502 comparisons strongly weaken the blanket account. | I1 proves some semantics matter; I1-T2 and the mutation-policy issue remain material. |
| Everything is generic replication-plus-mutation and there is nothing to explain. | Same nominal mutation process and successful reproduction coexist with sharply different retention; the copying operator supplies missing structure. | Whether that distinction adds predictive power for adaptation or only describes present sequence inheritance. |

### The origin-of-life statement I would defend

**An origin of replication need not be an origin of a rich hereditary channel.** A system may repeatedly reconstruct a successful form while suppressing the variations required for cumulative change. The origin-of-life question is therefore not exhausted by finding something that multiplies: one must identify which differences reproduction preserves, and how the surrounding chemistry selects among reproductive mechanisms.

This is a general *question and explanatory distinction* supported by a concrete experimental example. It is not a demonstrated historical sequence for Earth. It does not choose metabolism-first over genetics-first, prove that membranes are dispensable, or establish that a fetch boundary is a biological individual. The serious biological bridge is perturb-and-propagate experiments on chemical replicators or autocatalytic networks, not picturesque comparisons between instructions and molecules.

Small automata are useful for exposing the minimal mechanism; they are not automatically more general than machine code. At the finite tape sizes and budgets actually used, these are finite-state systems. Calling them little Turing machines does not make a cross-substrate universality argument. Compilation preserves computable functions under suitable assumptions; it does not preserve random-program measures, mutational neighbourhoods, reproduction costs or emergence times. The paper should abstract the operator, then test it in another realization.

## 3. Three competing paper architectures

These are genuinely different editorial choices. Do not average them into a manuscript with three introductions and three climaxes. The word budgets below include the abstract and exclude legends, references and Methods; they are planning budgets, not a claim about an editor's final allowance. Each can be compressed toward 3,000 words by reducing context and secondary interventions. Figure plans use existing evidence and explicitly label constructions, exploratory analyses and pending validation; they do not assume an unperformed experiment has succeeded.

### Architecture A — the origin of a hereditary channel (recommended)

**Working title:** The copying mechanism sets the capacity for heredity

**Story.** Build a microscope for reproduction, then show that the obvious readout was insufficient. A soup can become full of successful copiers without acquiring a comparably rich ability to preserve variation. Branching rules out simple lineage extinction. A common copying operation explains the difference between regenerating a pattern and transmitting sequence: its overlap sets the retained degrees of freedom. The population-level prevalence of those designs changes with the halting ecology. A second processor and controlled BFF interventions establish that the arrival route is contingent and that pointer confinement is not the general invariant. End on a concrete origin-of-life question: how does reproduction acquire a channel through which novelty can accumulate?

**Draft abstract.**

> Self-replication is often treated as the beginning of evolution, but producing copies need not preserve the differences on which evolution acts. We built Zilion, a validated GPU execution substrate, to follow replication and inheritance in interacting populations of short programs. In Z80 populations, early copying lineages persisted under branching transfer while losing the tested mutations. Later, reliable copiers divided into regenerators that restored a recurring pattern and transmitters that preserved unexecuted sequence. For a recurring copying architecture, an exact overlap law predicted the number of independently transmissible positions, linking a local memory operation to population measurements. Changing the halting environment altered the prevalence of the two designs. Experiments with a second processor and modified BFF languages distinguished this hereditary mechanism from a universal sequence of reproductive stages. These results identify a tractable problem for origins research: explaining how a process that reproduces acquires the ability to transmit variation.

**Five main figures.**

1. **An instrument that separates copying from inheritance.** a, Zilion's minimal execution contract and the actual encounter, with supplied boundaries/reset visible; b, three short animations rendered as a static sequence—partner-dependent copying, regeneration, sequence transmission; c, one measured Z80 population trajectory showing why abundance alone misses the distinction. The purpose is to establish the experiment, not display a software architecture chart.
2. **A lineage can persist while its variation disappears.** a, unmutated branching survival alongside mutant-allele retention, sharing the generation axis but not a misleading common denominator; b, pusher/regenerator/transmitter site-retention maps; c, the parent-by-environment matrix, showing extinction separately from surviving lineages with zero sites; d, one literal mutant-to-offspring trace that makes erasure visible. X1 is the decisive panel.
3. **Overlap determines what is inherited.** a, the same abstract copying operation at two displacements, with source bytes and their descendants; b, the d−4 prediction and empirical groups, showing world-level dispersion and the scope of eligible cores; c, constructed 6502 d−6 tests, explicitly distinguished from evolved 6502 populations. A one-byte displacement intervention is a mechanistic diagram, not mislabeled as the R/T ecological experiment.
4. **The environment changes which copier prevails.** a, matched L=32 benign/lethal mixture trajectories from both starting directions; b, population class distributions rather than the exact modal genotype; c, the lethality dial's transmitter trend and nonmonotone closure timing; d, measured encounter outcomes with third-party intruders. State on the figure that the full equilibrium mechanism is unresolved. No fitted curve pretending the rejected mutation model worked.
5. **Different routes, the same questions.** a, Z80 versus 6502 first-event histories with censoring, distinct stage definitions and the failed registered PC-loop criterion visible; b, BFF literal intervention and bandwidth intervention as separate comparisons; c, a small matrix of observed persistence, fetch confinement, context dependence and variation retention, with “not measured” cells. Do not fill missing BFF/6502 serial-retention cells from analogy.

**Main-text outline: 3,650 words.** Abstract 150; introduction and origin-of-life problem 400; Zilion and operational measurements 400; replication without persistent tested variation 650; overlap law and its empirical reach 700; ecology of regeneration/transmission 600; cross-substrate scope and discussion 750. Methods approximately 2,800 words: execution contracts and versioning 450, worlds/sampling 350, reproduction/retention assays 650, copy-law conditions and census methods 350, registration/statistics 400, validation/impact/reproducibility 600.

**Extended Data.** Full retention panels and threshold sensitivity; modal-versus-population detection; reset dependence and invasions; detailed ecology controls including failed predictions; genealogy synthesis/N6; short-program censuses; detailed S1 corrected outcomes; BFF variants; emulator/impact coverage. Combine related tables within the allowed display budget rather than hiding primary counterexamples.

**SI.** Complete registration history and component verdicts; unsuccessful explanations and their exact assays; all proofs; full genotype/encounter inventories; ancestry-definition amendments; software manifests and machine-readable data dictionary. Early ablation/size/padding atlas belongs mainly here or in Extended Data, not in three main Results sections. Detector failures deserve one Methods sentence and an Extended Data benchmark, not a separate main-text attack on biosignatures.

**Theorems.** Proposition 8's *claim and mechanistic derivation* occupy the main text, with full hypotheses and proof in Methods/SI. Lemma 7 appears only as the definition/scope of the context-dependence measurement in Methods. Theorems 1, 2 and 6 and Propositions 3 and 4 go to SI, with a finite-census result summarized in the scope section. Corrected Proposition 9 goes to Methods/SI; Conjecture 9′ remains explicitly open. Model 5 is an examined, incomplete model in SI, not the paper's main theoretical conclusion.

**Zilion.** Named in the abstract, introduced in paragraph three or four, visible in Figure 1, documented in Methods. Its semantic contract is part of the scientific method. Performance benchmarks and the general API are in the companion tool materials. The paper must disclose which historical experiments used Algocell kernels and which used the newer Zilion backend; a rewrite must not retroactively imply that every historical result was run through the same release.

**Editorial judgment.** This has the best center of gravity. It answers an old conceptual question with unusually inspectable mechanics. The danger is overselling a general distinction long recognized in origin-of-life work as if we invented it. The news is the experimentally evolved mechanisms, their exact retention law and ecological selection among them. A prospective test of the law on evolved 6502 copiers would materially strengthen it, but the present constructed result must not be disguised as that test.

### Architecture B — a comparative science of the beginnings of replication

**Working title:** Different computational substrates take different routes to replication

**Story.** Zilion makes the “digital chemistry” an experimental variable rather than a hidden choice. The Z80's rapid open pattern writers are one accessible solution, not the universal first organism. A short-program census guides a prospective 6502 comparison that begins with confined copiers and much later detected emergence. Controlled changes within BFF and Z80 expose the relevant operations and environmental constraints. The resulting account separates existence of a copying mechanism, its establishment and its capacity to carry variation. The central achievement is a predictive comparative method, with the copying law as evidence that some mechanisms transfer even when the chronology does not.

**Draft abstract.**

> Why do some environments readily produce self-replicators while others reach them by different routes? We developed Zilion, a validated GPU substrate for experiments across instruction sets, and combined exhaustive program censuses with registered population tests. Z80 populations commonly began with open pattern writers. A census of a second processor, the 6502, found no equivalent short exact self-writer; its first detected replicators instead remained within their own instruction region and appeared much later. Controlled changes to literal writing and halting in BFF and Z80 altered these beginnings, showing that an available copying operation alone does not determine the population outcome. We further distinguished reliable pattern regeneration from the transmission of arbitrary sequence, and related their difference to copying overlap. Together, the experiments replace a universal sequence of digital life stages with testable constraints on reproductive mechanisms. They establish an approach to asking which features of early evolution depend on the substrate and which can transfer between substrates.

**Five main figures.**

1. **One instrument, explicit alternative worlds.** a, common encounter schema; b, the independent axes—ISA, address map, reset, budget, halting—with actual tested settings; c, validation coverage and a concise indication of unresolved ecological impact. Avoid a visual claim that “128 instructions” is physically equal work on different machines.
2. **Census before population experiment.** a, exact self-writer counts by primitive period for Z80 and 6502; b, an immediate-load/push construction and the restricted width obstruction; c, the chronology of census, registration, pilots and S1, showing that Proposition 9 was formulated afterward; d, short constructed 6502 loop as a positive control.
3. **The second machine takes another route.** a, censored time-to-detection distributions; b, open/confined/loop counts with numeric-PC and ring-position measures separated; c, representative Z80 and 6502 traces; d, the full arm-A registered verdict strip, including P1 failed and P3 untested. The corrected result is compelling enough to display honestly.
4. **Interventions reveal the relevant constraints.** a, BFF with/without literal writer; b, literal bandwidth versus inflow, with pointer entry shown independently; c, Z80 lethality dial, showing route loss and the failed monotone closure prediction; d, random-stack-pointer alternative or ring-padding rescue. These panels prevent attributing everything to literal width.
5. **What transfers beyond the first event.** a, regeneration versus transmission as reproductive operators; b, Z80 overlap law plus constructed 6502 test; c, observed final class distributions and missing serial measurements; d, a compact map of claims: theorem, finite census, prospective observation, conjecture. This is the payoff of comparison, not a universal phase diagram.

**Main-text outline: 3,700 words.** Abstract 150; comparative question and lineage 400; Zilion and contracts 550; finite census and mechanistic hypotheses 600; registered second-machine result 600; within-substrate causal interventions 650; hereditary mechanisms and generality 450; discussion 300. Methods approximately 3,000 words, weighted toward execution semantics, exact enumeration, validation coverage, S1 corrections, censoring and registration chronology.

**Extended Data.** Complete S1 B–D and sensitivity tables; py65 exclusions; Z80/Q semantic-impact tables and I1-T2 verdicts when available; 6502 positive controls and near-copiers; BFF switches; reset/padding interventions; detailed retention and ecological competition. **SI:** censuses, proofs, full opcode coverage, all failed predictions and scorer history, performance and cost, complete genealogy and early atlas.

**Theorems.** The restricted literal-width argument, *after repair*, can be stated briefly in the main text, explicitly as a construction/obstruction for one grammar. Its proof belongs in Methods. Proposition 3 appears as a computed classification, not mathematical inevitability. Proposition 8 provides the main transferable law in Figure 5; full proof in SI. Theorems 1 and 2 supply resource/loop constraints in Methods or SI. Theorem 6 must be demoted to its conditional scope; Lemma 7 defines the measurement; Proposition 4 and Model 5 remain SI. Conjecture 9′ cannot bear the main conclusion.

**Zilion.** The named enabling technology and part of the headline contribution. A companion release is essential for this architecture. It needs a genuinely reproducible public pair experiment, not only a browser visualization of a soup.

**Editorial judgment.** This architecture best matches the technology-first arc and the user's wish for generality. It is also the most vulnerable to a referee saying “two processors, several changing conventions, and one failed principal criterion do not yet make a general theory.” An orthogonal synthetic-ISA experiment would improve it more than a third historical CPU. The present draft of Proposition 9 is not strong enough to serve as the decisive Nature-level theoretical surprise.

### Architecture C — how new copiers are made

**Working title:** How new replicators enter a replicating world

**Story.** Most evolutionary accounts follow organisms that already reproduce. Here we observe the birth of a new reproductive mechanism inside a population dominated by a different one. Exact encounter records show that sampled first confined founders are completed by transformations involving non-copying whole tapes, rather than by a final point mutation of an existing copier. The critical distinction is between material ancestry, the program that executes and the function that reproduces. A new registered cohort confirms the encounter-completion result while rejecting the stronger claim that the writers cannot copy even their executed cores. The paper argues for studying origin events as reaction paths before treating them as ordinary parent–offspring evolution.

**Draft abstract.**

> Evolutionary histories usually begin with entities that already reproduce. To study how a new reproductive mechanism originates, we recorded encounters and mutations in populations of short interacting programs. In Z80 worlds, open copying patterns preceded the sampled lineages of confined copiers. Across 37 worlds that established such lineages, no sampled first founder was completed by a point mutation; encounters rewrote or transferred the tape that became the founder. A registered replication confirmed the encounter-completion result and the intervening chains of whole-tape non-copiers. However, some producing programs already copied their executed cores, showing that the inferred origin depends on the reproductive unit being measured. We linked these histories to biased availability of component words and to the distinction between regenerating a pattern and transmitting sequence. The results make origin events experimentally inspectable while separating their actual causal history from claims that interaction is necessary, or universally superior, for discovering replication.

**Four main figures.**

1. **What is a reproductive origin event?** a, encounter/mutation recorder and exact replay; b, whole-tape versus executed-core copy criteria; c, sample-selection diagram from all worlds to sampled founder lines. Establish the inferential unit before displaying a spectacular ancestry tree.
2. **Two complete origins.** a–c, an open writer producing a founder, showing pre-state, executed instructions and product; d–f, a mutation-created producer that already copies its executed core. Their contrast makes the definition-sensitive result concrete and prevents the paper from depending on its most flattering example.
3. **The registered replication.** a, founder-completing events in discovery and N6 sets; b, delays and intervening whole-tape copy phenotypes; c, producer phenotypes and abundance null, with N6-4 marked failed; d, local mutational reach and executed-word enrichment. Do not resurrect the withdrawn event-kind P value.
4. **A new copier is not yet a rich hereditary channel.** a, first-founder functional type versus later population type where data allow; b, branching retention contrast; c, overlap mechanism; d, supplied versus evolved components of reproduction. The final panel establishes the paper's biological reach and its limits.

**Main-text outline: 3,500 words.** Abstract 150; origin versus ordinary descent 450; recording technology and reproductive units 550; two origin events 600; discovery/confirmation across worlds 700; consequences for inheritance 550; discussion and competing explanations 500. Methods approximately 2,900 words: ancestry and founder definitions 800, recording/replay 450, assays and negative controls 550, reach calculations 350, registration/amendments 400, worlds/statistics 350.

**Extended Data.** All-founder summaries; alternate ancestry/copy definitions; later conversions versus first founders; X3 scope and off-line confined tapes; withdrawn N5 path tests; additional allele-retention panels; 6502/BFF route comparisons as boundaries of the Z80 claim; emulator validation and impact. **SI:** full causal traces, raw record schema, all pipeline amendments, detailed counting and copying proofs, early atlas and ecological studies. The 6502 cannot be sold as replication of the genealogy finding: no corresponding 6502 genealogy experiment is in the record.

**Theorems.** No numbered theorem needs to appear in the main text. Proposition 8 can explain a reproductive product in prose; the local mutational-reach calculation belongs in Methods. Model 5 can motivate a future reaction-rate account, clearly unvalidated. Theorems 1, 2 and 6, Propositions 3, 4 and 9 and Lemma 7 belong in SI unless needed for an assay definition. The absence of a main theorem is not a defect if the causal histories do the explanatory work.

**Zilion.** Presented as the execution/replay substrate with the event recorder as the critical measurement system. Again distinguish the historical recorder's Algocell implementation from the newer generalized Zilion interface. A trace browser is the most valuable public artifact for this version.

**Editorial judgment.** This is the best narrative drama and the greatest risk of rediscovering what a program transformation obviously does. BFF's 2024 paper already traced a replicator's emergence, and the latest BFF critique challenges special claims about interaction-driven search. Our genuine increment is repeated, validated causal histories plus a frozen confirmatory cohort. The most exciting broader claim—interaction creates a qualitatively superior route into replication—has not been tested. I would not choose this as the main Nature submission on present evidence.

### Comparative rating

Scores run from 1 to 5. Higher is better except reviewer risk, where 5 means greatest risk. Generality rates the *current evidence*, not the ambitions of the prose.

| Architecture | Surprise | Generality | Hard to vary | Reviewer risk | Serious OOL reach without overclaiming |
|---|---:|---:|---:|---:|---:|
| A. Copying mechanism and heredity | 5 | 3.5 | 4.5 | 3 | 4.5 |
| B. Substrate-dependent routes | 4 | 4 | 3 | 4 | 3.5 |
| C. Birth through encounters | 4 | 2 | 3 | 5 | 4 |

**Choose A, borrow B's comparative discipline, and give C one excellent Extended Data figure.** Do not borrow all of B's and C's Results sections. A Nature-length paper cannot carry every discovery at equal volume. The fact that a result took three days of difficult work does not entitle it to three paragraphs.

## 4. Honest novelty assessment

### What the lineage already contains

The history should be a short argument about the evolving question, not a museum tour. Cellular automata established constructive possibilities for reproduction; Tierra and Avida made populations of reproducing programs experimental objects; artificial chemistries made the formation and organization of interacting computational objects central; modern soups made spontaneous replication in several executable substrates vivid and accessible. Our contribution should move from observing that reproduction happens to identifying the transformations that preserve or erase variation.

| Prior art, checked against primary sources | What it already owns | What remains distinctive here |
|---|---|---|
| Langton, *Self-reproduction in cellular automata* (1984). [Primary article](https://www.sciencedirect.com/science/article/pii/0167278984902562) | Simple self-reproduction need not require universal construction; a loop can carry a reproductive description. | Evolutionary discovery and quantitative retention under explicit execution interventions, rather than a first demonstration of a self-reproducing loop. |
| Ray, *An Approach to the Synthesis of Life* (1991). [Author's paper](https://tomray.me/pubs/alife2/Ray1991AnApproachToTheSynthesisOfLife.pdf) | Machine-code organisms, intrinsic reproduction, resource competition and emergent ecology from an introduced ancestor. | Studying the route from random material and measuring distinct hereditary mechanisms. “No externally imposed fitness function” is not itself a new idea. |
| Ofria and Wilke, *Avida* (2004). [Primary paper record](https://pubmed.ncbi.nlm.nih.gov/15107231/) | A controlled, instrumented experimental platform for reproducing and evolving programs. | A different encounter chemistry, explicit multi-ISA contracts and targeted tests of emergence and retention. Rigorous digital evolution did not begin with this project. |
| Fontana and Buss, *The Arrival of the Fittest* and *What would be conserved if the tape were played twice?* (1994). [Author's abstract](https://sites.santafe.edu/~walter/Papers/arrival.html), [companion paper](https://sites.santafe.edu/~walter/Papers/tape.html) | Constructive computational chemistry and the problem of how organizations and reproducible forms arise. | A low-level executable setting in which a local copying law and intervention assays can be connected to population outcomes. Do not claim the organization question was absent from ALife. |
| Agüera y Arcas and colleagues, *Computational Life* (2024). [Primary paper](https://arxiv.org/html/2406.19108v2) | Spontaneous replication across languages/ISAs, an explicit BFF emergence trace, zero poisoning, the Z80 stack-to-block-copy succession and the repeated `01 c5` mechanism. | Persistent allele-retention tests, the overlap law's empirical use, prospective 6502 comparison, systematic counterexamples and a stronger execution-validation contract. |
| Cicala and colleagues, *Coevolution of self-replication and function in a digital primordial soup* (2026). [Primary paper](https://arxiv.org/html/2607.09211v2) | Coevolution of reproductive architecture and rewarded functions; load–push/block-copy succession; compact copy cores and partner-offset interpretation. | Distinguishing regeneration from transmission without an assigned payload task and testing the quantitative overlap/retention relationship. Do not claim discovery of the core or of “room for useful code.” |
| Knierim, *BFF: Simple explanations for complex phenomena* (2026). [Primary paper](https://arxiv.org/html/2607.01483v1) | A mutation-only search can discover BFF replicators efficiently; lineage-merger limits can impede takeover more than discovery. | Our event histories concern the actual observed path in Z80. They do not rebut this result or establish that interaction is a superior search operator. |
| Vasas, Szathmáry and Santos, *Lack of evolvability in self-sustaining autocatalytic networks…* (2010). [Primary paper record and abstract](https://pubmed.ncbi.nlm.nih.gov/20080693/) | Self-maintenance/reproduction and evolvability can come apart; return to a built-in compositional attractor need not support cumulative evolution. | An independently evolved, byte-resolved example with an exact copying operator and ecological switches. The distinction is not new; the mechanistic realization and tests may be. |
| Dalton and colleagues, *Accelerating Reinforcement Learning through GPU Atari Emulation* (CuLE, 2019/2020). [Primary paper](https://arxiv.org/abs/1907.08467) | Parallel GPU emulation is established methodology. | A substrate designed for mutable tiny programs, configurable digital chemistries, ISA comparison and scientific provenance. GPU emulation alone is not Zilion's novelty. |

This is a targeted novelty check, not a completed systematic review. It is enough to reject “first spontaneous digital replication,” “first GPU emulator,” “first open-to-closed succession,” and “first distinction between replication and evolvability.” Before submission, search the exact overlap/retention result and independent 6502-soup implementations more exhaustively. The planning research already notes external 6502 activity; claiming first use of that processor would be unnecessary and fragile.

The neural-network connection should be historically careful. McCulloch and Pitts' formal neural model dates to 1943 and connects neural activity, logic and computation. That supports a shared intellectual ancestry in mathematical biology and cybernetics, not a simple claim that today's neural networks descended from the later field called ALife. One sentence is enough; it should not serve as borrowed prestige. [McCulloch and Pitts, primary paper](https://doi.org/10.1007/BF02478259)

### Methodological innovation: substantial integration, moderate component novelty

The ambitious part of Zilion is not that a GPU executes a CPU. It is that execution semantics become recorded, manipulable experimental factors across instruction sets, with small-world address effects and self-modifying programs treated explicitly. Coupled to scalable assays, censuses and validated event records, that is a valuable scientific instrument. Reusable cross-ISA experiments and their provenance can be more important to the community than a peak-throughput number.

Registration, differential testing, exact enumeration, multi-seed experiments and independent reimplementation are individually established methods. Their disciplined combination improves the credibility of an unusually easy-to-misread system. Claim rigor and reusable capability, not priority for scientific hygiene. The many failed predictions are evidence of actual discrimination only if the final manuscript preserves the original verdicts and separates later explanatory repairs. A preregistration repository is a chronological record, not immunity from endpoint mistakes.

The strongest methodological weakness is that validation is uneven across layers. An opcode reference test, a pair-memory comparison, a retention assay and an ecological rerun answer different questions. The record now recognizes this, especially through I1/I1-T2 and S1 corrections. That layered view should be one of the tool's lasting contributions.

### Mathematical innovation: elementary tools, potentially strong explanation

I would rate the current mathematics **low in technical depth, moderate in conceptual organization, and high in usefulness for the overlap mechanism**. Pigeonhole, expectation counting, periodicity and conditional entropy are not novel mathematical machinery. Calling all of them theorems creates a misleading impression of depth and makes the main paper harder to read.

Proposition 8 matters because it replaces a loose biological metaphor with an exact intervention prediction. Proposition 9 could add a concise classification insight after repair, but its reach is narrower than the planning prose. Neither supplies an arrival-rate theory. The genuinely deeper programme is a causal theory of reproductive channels and a nonequilibrium theory linking encounter kernels to establishment and hereditary capacity. That programme is not yet a result of this paper.

### Results: where the surprise really is

The highest-surprise results are X1's survival-without-retention, regeneration versus transmission within a recurring mechanism, the quantitative copying law's fit, and the shift in their ecological prevalence. The best generality result is the prospective second-processor comparison, including its honest failed loop criterion. The strongest methodological discoveries are the detector/class-marker failures and the realization that copying should be tested over descendants, not inferred from a single tape.

The open-to-closed succession itself is incremental. The genealogies are unusually rich but definition-sensitive. The literal BFF intervention is causally persuasive but engineered to introduce precisely the expected mechanism. The 6502 census is impressive finite coverage; it becomes conceptually important through the prediction it informed, not through billions of evaluated strings alone.

**Strongest single claim I believe we can defend:**

> In spontaneously arising digital replicators, reliable reproduction can either erase or transmit sequence variation; in a recurring copying architecture this difference is quantitatively determined by overlap, and the execution environment changes which kind predominates.

Scope immediately follows: the exact law concerns specified copying families, the strongest persistent-retention evidence is Z80, the 6502 extension is constructed rather than demonstrated across evolved genomes, and the environment-to-equilibrium explanation remains incomplete. That scope sharpens the claim. It need not drain its character.

### A specific correction to Proposition 9

The draft assumes a common window of length at least `ℓ+w−2`, with `ℓ=w+2`. Fine–Wilf requires `ℓ+w−gcd(ℓ,w)`. For odd w, the required length is `2w+1`, while the draft supplies `2w`. The inference to period one therefore does not follow. The primary periodicity literature states the gcd-dependent bound. [Halava, Harju and Kärki, *The theorem of Fine and Wilf for relational periods*](https://www.numdam.org/item/10.1051/ita%3A2008025.pdf)

There is a concrete failure of the short-window inference: let w=1, distinct opcodes a and b, and literal b. The executed periodic stream is `(abb)∞`, while the written stream is `b∞`. They agree on a two-byte `bb` window, but a≠b. This is **not** a counterexample to the actual 16-byte 6502 census or a demonstrated 16-byte self-replicator. It is a counterexample to the inference the stated window condition licenses.

A sufficient repair of this step is to require at least `2w+2−gcd(2,w)` copied bytes. The 16-byte setting with w≤7 still satisfies that corrected requirement. For the full construction, also state the coverage, layout, budget and execution assumptions that ensure the proposed stream is actually written and executed as assumed. The theorem addresses an identical repeated load/push pair; it cannot silently become a theorem about arbitrary instruction mixtures, all open replication or which route wins a population race.

Finally, the chronology matters: **the census informed the preregistered S1 prediction; Proposition 9 was written after S1.** The paper may say the theorem explains the census retrospectively. It should not say the theorem prospectively predicted the 6502 outcome. The 8086 witness in the planning material is a useful next construction, not a completed third-processor population experiment.

## 5. Theory programme ranked by explanatory reach

The ranking is by what would explain the most and transfer furthest, not by what is easiest to publish as a theorem. The first three directions form one coherent programme: identify what reproduction preserves, derive it from the copying operation, then predict which operations the ecology sustains.

### 1. Reproduction as an interventional information channel

**Question.** Which distinctions between parents remain distinguishable among their descendants, after accounting for environmental reacquisition and lineage extinction?

**Proposed object.** Choose a declared ensemble of interventions A on a parent, independently sample contexts E, and record descendants at generation G, including an explicit extinct outcome. Measure how well descendants identify A. Report both the total channel and a clearly labeled conditional-on-survival result; conditioning alone can manufacture an apparent improvement by selecting the easiest surviving cases. In branching populations the output is a population, not one arbitrarily chosen child. Distinguish information conveyed through descendant material from information continually supplied by neighbours.

**Provable.** Data-processing constraints for genuine serial Markov channels; erasure results when all interventions map to the same output law; finite-channel bounds; sufficient conditions for preservation on an invariant family. These must be stated with their independence assumptions—partner pools correlated with the parent can inject information, and selection/reacquisition can invalidate a naive serial channel model.

**Needs a model.** Relating this channel to adaptive potential requires reproduction, mutation, selection and a declared class of ecological challenges. Mutual information alone does not equal fitness or open-ended evolvability. The presently reported number of transmissible sites is an assay statistic, not already a Shannon capacity.

**Needs experiments.** In-soup interventions with ancestry-aware tracing, matched partner pools, neutral tags that do not supply their own copying mechanism, longer branching histories and the same assay on evolved 6502/BFF copiers. The decisive challenge is to retain a new difference through changing environments rather than repeatedly reacquire it from a fixed pool.

**Reach.** Applies to program tapes, cellular automata patterns, compositional chemical assemblies and template polymers. It answers the origin-of-life question without first imposing a genome/phenotype division. This is the best long-term theoretical investment.

### 2. A general theorem of copy dependencies and retained degrees of freedom

**Question.** What features of an execution make reproduction a projection, a permutation, a duplication or an erasure of parental material?

**Provable.** For a fixed admissible execution consisting of copy assignments, unroll the writes in time. Each final coordinate traces to an initial coordinate or a constant. The number of distinct unconstrained source coordinates reaching the offspring bounds the number of independently preservable symbols. Under independent alphabet-valued inputs and exact copying, r such coordinates yield at most `|alphabet|^r` distinguishable outputs, with equality when all are observable and viable. Repeating the execution induces a dependency map whose surviving components identify persistent coordinates. Proposition 8 is a special case, not the entire theory.

The assumptions matter: control flow can change when an input changes; arithmetic can combine symbols; viable inputs may not form an unconstrained product set. A theorem for all self-modifying programs cannot simply reuse the rank of one observed trace. Start with declared copying grammars, then generalize to piecewise maps or symbolic execution.

**Needs a model.** Approximate copying, multiple interacting templates, mutation and context-dependent branch choices. A useful model should predict which departures from the exact law occur, not merely classify failures as “outside the theorem.”

**Needs experiments.** Matched displacement/payload interventions; the correctly measured displacement in evolved 6502 copiers; an unrelated synthetic automaton that implements the same dependency structure. Test held-out copying designs, including reversals, stride copies and non-dividing offsets.

**Reach.** Stronger than literal-width parity because it characterizes the reproduction operator rather than a particular instruction encoding. A chemical analogue would track which molecular distinctions template or catalytically determine descendant distinctions; it must allow that chemistry does more than literal coordinate copying.

### 3. A nonequilibrium theory of ecological selection on heredity

**Question.** Why does one halting rule sustain regenerators and another sustain transmitters, when direct R/T encounters are nearly balanced?

**Needs a model first.** Treat an encounter as an empirical reaction, `(i,j) → (i′,j′)`, with its measured outcome distribution, and include mutation and execution scheduling separately. A minimal population equation sums production/loss from those reactions. Crucially, retain at least the core-free intruder classes that the current two-type explanations discard. If local mixing matters, track pair correlations or explicitly justify a well-mixed approximation.

**Provable within the model.** Invasion criteria, stationary points, stability, perturbation responses and bounds on when a coarse-graining preserves the relevant observables. In a fixed environment, the spectral radius of a correctly constructed next-generation matrix can separate expected lineage growth from extinction; it does not by itself measure the preservation of particular variants.

**Needs experiments.** Estimate encounter and mutation kernels on one set of populations, then predict mixtures and invasions in held-out worlds and new halting probabilities without fitting their equilibrium. Measure the terms that currently fail to account for benign-tar advantage. Compare predictions at more than one spatial mixing rate.

**Why statistical mechanics belongs here.** This is an interacting stochastic many-body system with ongoing mutation and execution. Master equations, coarse-graining, correlation functions and first-passage methods can explain its dynamics. A Gibbs distribution or free-energy minimization does not follow merely because the system is large. Detailed balance has not been established. “Mutation is temperature” is an analogy until a mapping is derived.

**Reach.** Converts an ecological observation into a predictive explanation and links sequence-level mechanisms to population outcomes. It also supplies the correct biological claim: heredity capacity can hitchhike with immediate ecological advantages, without selection anticipating future evolvability.

### 4. A theory of founder supply and establishment

**Question.** How do substrates and existing populations change the rate of producing *established* new reproductive mechanisms?

**Needs a model.** Separate candidate production from establishment. An encounter-mediated founder hazard has contributions schematically like `Σ n_i n_j k_ij→F p_est(F | environment)`, with the appropriate normalization and spatial pair frequencies. Mutation contributes another term. The original `q∫n dt` window is a special approximation in which both production and establishment are compressed into q; those quantities need not stay constant during ecological change.

**Provable within the model.** Survival/first-arrival distributions under explicit stochastic assumptions, comparisons of competing hazards, and conditions under which a large pre-replicator population accelerates or instead inhibits a later mechanism. Nonmonotone closure time can arise if an intervention simultaneously changes precursor supply, destructive encounters and establishment. That is a candidate explanation of DZ, not an established fit.

**Needs experiments.** Hold founder genotype and introduction count fixed to measure establishment; separately manipulate candidate-producing tapes and their substrates; compare mutation-only and interaction-enabled search under matched evaluated-program and execution budgets. Trace origins on 6502 and under lethal Z80 conditions rather than extrapolating the L=16 benign ancestry result.

**Reach.** Addresses Fontana's arrival problem directly and unifies census, genealogy, ecological intervention and the distinction between discovery and takeover. It could become a deeper main-paper theory if it predicts a new route or waiting-time distribution prospectively.

### 5. Minimal copying grammars and representation-sensitive complexity

**Question.** Which primitive operations permit a self-writer, at what minimum length, with what context dependence and retained degrees of freedom?

**Provable.** Repair Proposition 9, precisely define its stream/coverage assumptions, and attempt a proof or counterexample for Conjecture 9′. Use bounded model checking or exhaustive symbolic enumeration to obtain complete statements for finite programs, budgets and contexts. Finite searches through length 64 would still be finite evidence, not a proof for all lengths.

**Needs a model.** A general “replicative complexity” measure must specify the encoding and resources. Program length alone is not invariant under changing the instruction set: an instruction can package an entire copier. Better comparisons fix primitive operations, locality, storage and budget, or account for compilation overhead explicitly.

**Needs experiments.** A matched family of small machines varying literal width, write coverage, address locality and stopping independently. This is more informative than comparing Z80 and 8086 and attributing all differences to one architectural trait.

**Reach.** Useful TCS that identifies obstructions and constructive possibilities. It does not turn universal computability into a universal distribution of evolutionary outcomes. The biology-facing result should be a statement about required transformations, not an odd/even rule for a human-selected byte encoding.

### 6. Reproductive units and causal ancestry

**Question.** What is the reproducing entity when a tape copies its executed core but not its unused tail, or when a copier is produced by another program?

**Provable or definable.** A clear distinction between material provenance, execution causation and functional transmission. Represent encounters as a causal hypergraph if two inputs contribute; do not force a unique parent where the mechanism has none. Define a candidate reproductive unit through its intervention response across descendants rather than through its share of ancestral bytes alone.

**Needs a model.** Which coarse-grained unit supports a stable, predictive parent–offspring relation? A dynamical unit may be a sub-tape, a set of interacting tapes or an encounter cycle. Any inferred boundary must be tested against the experimentally supplied tape/turn boundaries.

**Needs experiments.** Perturb core and tail separately, disable particular writers while holding the material environment fixed, and test whether an alleged multi-component unit needs all of its components for continued reproduction. Trace-first approaches cannot establish causal necessity without such interventions.

**Reach.** Directly relevant to autocatalytic networks and protocells. It would make “individuality” an earned result rather than a synonym for the instruction pointer staying on one side of an imposed boundary.

### 7. Error thresholds for the variation that is actually transmitted

**Question.** When does mutation overwhelm a hereditary channel, as distinct from merely eliminating exact modal genotypes?

**Provable within a declared model.** Standard mutation–selection operator results can be specialized to the transmitting subspace, with erasure by reproduction distinguished from random copying error. Regeneration can correct a pattern while destroying an introduced allele; this is not the same mechanism as an excessive external mutation rate.

**Needs experiments.** Mutation sweeps measuring persistent parental distinctions, functional class abundance and lineage growth separately, including explicit accounting for mutation attempts versus realized byte changes. Repeat across retained payload sizes and environments.

**Reach.** Valuable bridge to origin-of-life limits on sequence inheritance. Lower priority than the first three directions because applying an error-threshold formula before identifying the reproductive channel would redescribe the problem rather than explain it.

### 8. Are the apparent transitions genuine phase transitions?

**Question.** Are the abrupt changes in emergence, survival or class mixture a sharp transition, a finite-population crossover, or a detector threshold?

**Needs a model and experiments.** Specify an order parameter and a scaling limit; vary population size, spatial scale, observation horizon and mutation influx. Measure fluctuation and correlation scales, hysteresis where meaningful, and finite-size dependence. Persistent mutation can prevent a strictly absorbing dead state, so an absorbing-state universality class cannot be imported without checking its assumptions.

**Provable only after that specification.** Results about the selected stochastic process, not the claim that all computational soups share a known universality class. The present evidence establishes striking regime changes, not critical exponents or a thermodynamic phase transition.

**Reach.** Potentially substantial, but currently below the mechanistic programme. “Edge of chaos,” “dissipative adaptation,” Landauer cost and entropy production should not decorate this paper. GPU electricity use is not an organism-level energy model; physical dissipation requires a declared implementation and reservoirs. The vision document's thermodynamic samplers are a separate engineered probability model, not automatically the law of the evolving soup.

### What I would do before adding more mathematics to this paper

First repair Proposition 9 and finish/scoring-read I1-T2. Then prioritize **one held-out test of the copy-dependency law on evolved 6502 genomes**, using a correct mechanistic displacement measure and multi-generation retention. If that cannot be made clean, a matched synthetic-automaton test is a more interpretable extension than another loose CPU comparison. Separately, the most consequential new empirical question is whether transmitted payload variation can support a selected function across changing contexts; the current unexecuted bytes demonstrate an available channel, not a demonstrated cumulative evolutionary use of it.

None of these recommendations is permission to rewrite a registered verdict or quietly extend a horizon until it passes. They are proposals for new tests. The current paper can make its central distinction without waiting for a universal theory of life.

## 6. Zilion's role and the community programme

### In this paper: the instrument, with a scientific contract

Zilion should have a name, one clear introductory paragraph, a role in Figure 1 and a substantial Methods account. Its central promise is: **the experiment specifies what a program can do, and a second implementation can check that it does it.** ISA, address wrapping, stack initialization, budget units, halting and reset are parts of the model. They cannot remain incidental details buried in a shader.

The method should explicitly distinguish four checks: instruction semantics, one-encounter behavior, isolated reproductive assays and ecological outcomes. I1 shows why passing one layer does not settle the next. The same distinction makes the tool useful well beyond this project.

The paper should not read like a product launch. It needs enough validation to make a skeptical reader trust the experiment and enough access to let that reader challenge it. It does not need the full API or the vision's eleven learning branches. Current features and roadmaps must be visibly separate: the architecture describes mature and experimental cores alongside work still planned, including a general CPU twin and further batching/storage facilities. BFF's backend/export status also needs a version-specific statement.

### A separate tool paper: yes, as a coordinated companion

**Recommend a companion Zilion methods/tool paper and release, not a second broad Nature-style story.** Its contribution would be a reusable execution substrate for comparative artificial-life experiments, demonstrated by at least Z80, 6502 and the implemented BFF path, with clear limits on each. Publish performance as useful throughput at a declared semantic workload, including executed-step counts and halted lanes; an equal nominal instruction budget can hide different work.

The strongest evaluation is that an independent user can add a new world or mechanism test without reproducing this project's kernel archaeology. A release suitable for that claim needs stable manifests, a reference corpus, readable trace examples and a clear separation between core semantics and ecological policy. Community use and an independent reproduction would strengthen a tool paper more than another speed record.

Do not claim Zilion itself supplies a universal chemistry, nor that changing processors while keeping an API fixed keeps the experimental physics fixed. Those choices become observable and comparable; that is enough.

### Datasets: publish counterexamples alongside successes

The highest-value public collection is a **benchmark of reproductive mechanisms and their failure cases**, not just a large directory of world snapshots. It should contain:

- Frozen representatives of open copying, regeneration, transmission, unstable apparent closure and non-copying writers, with exact execution contracts.
- Partner sets, mutant interventions, multi-generation outcomes and explicit negative/background controls; include the cases that fooled similarity, entropy and motif detectors.
- The two-machine short-word censuses and positive controls, including near-misses and unsupported-reference categories.
- Validated founder traces, the distinction between discovery and confirmation sets, and the relevant corrected definitions.
- Original and corrected emulator/core versions, I1 pair differences and I1-T2 component verdicts when actually available.
- A machine-readable claim ledger binding each published number to the versioned data and scorer, with failed and untested components preserved.

Archive a citable release with checksums and a compact reproduction route for every main figure. Raw bytes without a semantic contract are not a reusable organism collection.

### An agent skill: useful interface, not scientific authority

**Recommend an accompanying agent skill after the contract and benchmark are stable.** It should teach an assistant to specify a world, select a validated core, run a known positive and negative control, register a new comparison before its outcomes, and regenerate the requested report. Its highest-value warnings come from this project's actual failures: motif identity is not functional identity; one-step copying is not retention; seed identity is not whole-world replay; a finite census is not an unrestricted theorem.

The skill should expose ordinary reproducible commands and cite the relevant manifests. It should not act as an oracle that decides whether a pattern is alive or retroactively repairs a failed prediction. An agent can make the experimental machinery accessible; the result still needs a declared question, public criteria and an independently checkable record.

### Browser reproduction: a scientific demonstration, not just a spectacle

**Recommend two browser experiences.** First, a deterministic encounter bench: select a frozen copier, alter one byte or the copy displacement, choose a partner, execute and watch the retained/erased distinction propagate through descendants. Show the exact contract and export the small result. This is the browser artifact most directly supporting the preferred paper.

Second, a founder-trace viewer with preserved recorded events and replay, plus an optional small stochastic live soup. Clearly distinguish replay of an archived record from a new population run. Exact historical Z80 trajectories cannot be promised from seed alone across devices; statistical replication and deterministic pair replay are different deliverables. The full GPU-scale census and confirmatory cohorts remain archived computational artifacts, not something a browser demo must recompute on opening.

The vision document's most useful principle is to build the measurement system alongside the organism. Its claims about multicellular-like organization, criticality and future learning machines remain research ambitions here. This paper should give that programme a trustworthy foundation, not borrow results from its future.

## 7. Questions for the user and for Claude

### For the user

1. **Which discovery would you most regret losing from the first page: the origin of persistent variation, the different beginnings on different substrates, or the encounter histories?** I recommend the first; this is a choice of scientific emphasis, not just a title.
2. **Do you want this paper to establish the availability of a hereditary channel, or its actual use for cumulative adaptation?** The present evidence strongly supports the former. The latter needs a functional, evolving payload result and would change the required work.
3. **Are you comfortable retiring “individuality” as the main claim?** We can retain it as a motivated future question while making the measured facts stronger and cleaner.
4. **What will the intended release make independently executable?** A validated pair benchmark, a figure reproduction bundle and a browser encounter bench would make Zilion's role concrete. A commitment to those is more valuable than a broad technology promise.
5. **How should we time the narrative freeze against I1-T2?** The final submission should incorporate the actual component verdicts; this independent synthesis cannot assume their outcome.
6. **If one additional generality experiment is possible, is there appetite for a prospective evolved-6502 retention test or a matched minimal automaton?** I favor whichever permits a clean prediction from the copy-dependency law over whichever sounds most impressive as a new machine.
7. **What ambition in the vision is essential to preserve in this paper's character?** I would preserve the ambition of making artificial life experimentally explanatory. I would leave competitive learning machines and open-endedness to results that directly test them.

### For Claude in Round 2

1. What is your strongest single-sentence claim after crediting the 2024 paper for both load–push mechanics and stack-to-block-copy succession? Which evidence would falsify your sentence?
2. Does your proposed narrative distinguish population persistence, execution confinement, partner independence and retention of parental interventions? Where does it require an implication between them that the data do not establish?
3. Can you find a stronger, hard-to-vary explanation of the R/T ecology than the incomplete encounter-demography account? If not, should ecology be the main climax or the opening of the next question?
4. Audit my Proposition 9 criticism: is the odd-width Fine–Wilf window indeed insufficient, and does the corrected bound plus explicit coverage assumptions establish precisely the restricted claim we want? Is there a counterexample to the arbitrary-mixture conjecture?
5. How much biological novelty remains after Vasas–Szathmáry–Santos and von Neumann's distinction between interpretation and copying of descriptions? I think the answer is the evolved mechanism, its quantitative law and ecology, not discovery of the conceptual distinction itself.
6. Would you lead on 6502 despite S1 P1 failing and P3 being untested? What exactly is prospective: the finite-census-guided prediction, the loop mechanism, or the later width theorem?
7. What is the fairest response to Knierim? Our genealogy data establish actual encounter completion, but not superior search efficiency or necessary recombination. Do you disagree, and on what comparison?
8. Which three currently main-text results would you remove entirely from the main narrative? My first candidates are the detailed size/ablation atlas, compression/assembly detector benchmarking and the full genealogy account.
9. Is the payload channel biologically compelling without a demonstration that its variation supports selected function? If you think it is not, identify the smallest new experiment that would change your view.
10. Which conclusions are conditional on the ecological reruns under corrected Z80 semantics, and which rest on a local mechanism or a second substrate independently? Please keep that partition explicit rather than treating “emulator validated” as a single global verdict.

### My position going into Round 2

The most powerful paper is about **how a process that copies becomes a process that can remember differences**. It should begin with a technology that lets us ask that question precisely, and end with an experimental programme that chemistry and other computational substrates can actually share. Confinement, literal channels, founder encounters and ecological competition earn their places by explaining or bounding that question. They do not all need to be co-headlines.
