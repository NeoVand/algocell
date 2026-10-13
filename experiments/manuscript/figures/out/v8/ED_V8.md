# Extended Data v8

Nine numbered displays, titles and PDF targets match the SI_v8.md Extended Data display map. No title mismatch remains. Sources below resolve from `experiments/`. All quantities use frozen numerical tables. Full ledgers and proofs accompany the displays.

**Unavailable measurements:** Applied-mutation/collision counts are unavailable for the Stage-L worlds. Pointer randomization was not measured and was not an 8086 S2 arm. The standard BFF harmless-bracket variant has no archived individuality-entropy panel; it appears in detection counts. These absences are stated in ED1, ED6 and ED7, respectively.

## Extended Data 1. Validation and correction impact

[PDF](ed8_1.pdf) · [PNG](ed8_1.png)

**Legend (180 words).** a, Validation comparisons identify their observational units and agreement criteria: instruction-vector states, ring conditions, isolated-assay metrics, claim verdicts and component outcomes. All 10,083 hardware divergences are expected. b, A ring condition is stable when the upper bound of its two-sided 95% Clopper–Pearson interval for the fraction of pairs with changed final memory is below 0.01%, using 100,000 encounter draws per condition. Flags indicate failure of that criterion. Core C0, core C1 and core C2 identify emulator implementations, distinct from displacement experiment C1 and 6502 retention experiment C2. c, Nineteen of 22 I1-T2 claims retain every constituent verdict, including prior failures. d, Four of 36 components change; applicability changes are distinguished from newly met criteria. e, Counts of differing measurements use 417 isolated assays per core comparison; rows overlap. f, All ten Stage-L worlds at each tape length complete 10,000,000 updates and are scoreable from retained final bytes and journals, although their 7,500-update snapshots are missing. Applied-mutation and collision counts are unavailable for these worlds. Expected hardware divergences and stochastic rerun differences do not isolate a causal effect of core repair.

**Sources:**
- `manuscript/figures/out/v8/data/validation_fractions.csv`
- `results/impact/outcome.csv`
- `results/tier2/T2_CLAIMS.csv`
- `results/tier2/T2_COMPONENTS.csv`
- `results/impact/assay_columns.csv`
- `results/tier2/T2_StageL_status.csv`
- `results/impact/supplement_manifest.json`
- `results/tier2/stageL_diagnosis.json`

## Extended Data 2. Survival, carriage and erasure

[PDF](ed8_2.pdf) · [PNG](ed8_2.png)

**Legend (215 words).** a, Historical serial transfer-eight outcomes over all identifiable mutant starts: lost, alive without the mutant marker window, or alive with it. Each serial population contains one descendant. b, Registered branching generation-eight outcomes distinguish loss, survival with fewer than half of members carrying the marker, and survival with at least half carrying it. The marker is the centred cyclic five-byte window of the introduced mutant. A scored site requires at least half of identifiable tested values to have an excess carrier share of at least 0.5: the fraction of surviving mutant populations classified as carriers minus the corresponding fraction of surviving unmutated populations. Serial populations use the same criterion with one member. c, Complete serial site maps at transfers one and eight. d, Carrier lineages divided by all mutant starts for six L16 parents across partner environments; each design has 65,280 mutant starts and 256 unmutated controls. Benign zero opcodes execute; lethal zero opcodes halt. e, Identifiable alternative types and unmutated starts for the serial designs. f, One block-copy L32 lineage against all-zero byte partners, through two 128-execution-step encounters. At step six, a read from address four carrying byte-zero ancestry overwrites the final byte-eight marker descendant. Survival requires at least 75% immediate-parent byte match at the best cyclic shift; marker carriage and ancestry are separate measurements.

**Sources:**
- `manuscript/figures/out/v8/data/serial_summary.csv`
- `manuscript/figures/out/v8/data/serial_sites.csv`
- `manuscript/figures/out/v8/data/branching.csv`
- `manuscript/figures/out/v8/data/parent_environment.csv`
- `manuscript/figures/out/v8/data/erasure_ordered.csv`

**Environment IDs for panel d:**
1. closed benign world
2. lethal world
3. open phase (t5000)
4. random partners, benign
5. random partners, lethal
6. random registers

## Extended Data 3. Inherited displacement and capacity

[PDF](ed8_3.pdf) · [PNG](ed8_3.png)

**Legend (196 words).** a, All 101 registered displacement experiment C1 settings in 22 Z80 parents, including 22 original controls. Free byte positions admit arbitrary alternatives jointly inherited with their causal roots while the core is fixed; markers coincide at eight coordinates with setting counts shown. b, All direct single-site, joint-payload and erased-coordinate challenges, separately for forward and reciprocal passages, total 1,067,608 cases without mismatch; each case uses 17 partners. c, Actual generation-eight descendant bytes and original-parent ancestry for parallel byte-zero interventions 0x04 and 0x60. d, The exploratory Z80 L32 first-byte sweep partitions all 256 values into 157 culture-negative, 88 culture-positive but theorem-ineligible and 11 culture-positive eligible values. Culture positivity requires dimensionless generation-two gain of at least 0.3 on 16 uniform partners; gain averages the increase in parent-byte match divided by available match headroom. e, The exploratory 5E ED B0 motif-bearing cell cohort from stages G, I, K and L; extra fetched-site scores and the immediate-parent whole-tape-match gate deficit are distinct outcomes. f, Frozen eligibility accounting for those 2,347 cells, and zero mismatches among 25,856 registered descendant-transfer records. Exact bytes, fixed core and ancestry each use that serial denominator. Eligibility and sampled scores do not certify a universal inheritance law.

**Sources:**
- `manuscript/figures/out/v8/data/capacity_edits.csv`
- `manuscript/figures/out/v8/data/ed_joint_challenges.csv`
- `manuscript/figures/out/v8/data/lineage_strips.csv`
- `results/capacity/serial.csv.gz`
- `manuscript/figures/out/v8/data/dial_coordinates.csv`
- `manuscript/figures/out/v8/data/z80_payload.csv`
- `results/capacity/candidate_eligibility.csv`

## Extended Data 4. Sampling, reset and classification

[PDF](ed8_4.pdf) · [PNG](ed8_4.png)

**Legend (185 words).** a, A separate random-register-reset cohort of 20 Z80 L16 worlds, with six observation updates. Confinement fractions use sampled heritable cells as denominator; two worlds are majority confined at 3,000,000 updates. b, Final-world modal-tape abundance versus modal identity among sampled heritable cells in the 170 worlds of panel c. c, Sampled-cell and world denominators by stage and tape length; confined and transmitter cells are subsets of the pooled cell rows. d, The Z80 L50 matched-pair invasion cohort comprises three worlds and 24 samples of 512 cells. Axes compare confinement among cells with and without the 20 F0 sequence marker; seed one at update zero has five of five marker cells confined and zero of 507 nonmarker cells confined. e, Exploratory order of first regenerator and transmitter detections at any positive estimated share. f, Complete sensitivity to majority and greater-than-10% class shares among heritable cells. Both-detected worlds equal the sum of the three mutually exclusive order columns. Regenerator and transmitter here mean confined site-score classes with at most two and at least five sites, respectively; they differ from competition byte classes. Observation order does not establish ancestry.

**Sources:**
- `results/r4/r4_rep.csv`
- `results/population/population_worlds.csv`
- `results/population/classes_cells.csv`
- `results/marker_check/marker_check.csv`
- `results/ladder/population_groups.csv`

## Extended Data 5. Environmental competitions

[PDF](ed8_5.pdf) · [PNG](ed8_5.png)

**Legend (201 words).** a–d, Historical core C0 and corrected core C1 Z80 L32 competitions under benign and lethal zero-opcode chemistry with mutation on, through 300,000 updates. Thin curves show individual worlds; thick curves show within-start-group medians. Each environment has ten 50:50 worlds and five worlds per reciprocal 1% invasion. R and T are partial-copy and full-tape byte classes. e, Complete endpoint medians for 110 worlds per implementation, covering both tape lengths, chemistries, mutation settings and starts. Mutation-off endpoints are at 100,000 updates. T share is T/(R+T); core share counts 5E ED B0-bearing cells over 20,000 cells; tail types count distinct sequences at bytes four through L−1 per world. The core-C1 L32 benign T-at-1% median uses four defined ratios among five worlds, while core and tail medians include all five, including zeros. f, Exploratory historical-core L32 continuations report mean T share at start and 2,000 updates. The lethal L32 host control reports unchanged-host fractions over 400 hosts times 64 soup-executor encounters per treatment. Detoxification replaces payload 0x00 and 0x76 with non-toxic bytes. Registered E1 under core C1, benign chemistry and mutation off meets its final-share interval in only four of ten worlds at each length, below seven required. Variable endpoints do not establish neutral drift.

**Sources:**
- `manuscript/figures/out/v8/data/competition_trajectories.csv`
- `results/offset/offset_switch.csv`
- `results/tier2/scored/E/offset_switch.csv`
- `results/ladder/n2_groups.csv`
- `results/toxin/causal.csv`
- `results/n2/demography.csv`

## Extended Data 6. Machine laws and finite censuses

[PDF](ed8_6.pdf) · [PNG](ed8_6.png)

**Legend (206 words).** a, Complete finite 6502 census accounting by tape length, BRK contract, execution-step budget and tiled-word period, including the four-billion-word period-four arms. Exact self-writers produce exact output into an all-zero partner. b, Complete finite 8086 census counts under the declared reset; nonzero words qualify by exact output into zero partners or at least 75% parent-byte match in at least eight of 16 uniform-partner encounters. c, The 296 constructed copier/contract configurations coincide at five coordinate groups. A payload position is carried when its parent byte remains at the same offspring position for all three sampled bodies and 64 partners per body; this is distinct from a mutant-value site score. d, S1 first-route counts for four 20-world 6502 arms; open and confined are fetch classifications, whereas loops overlap routes. e, The complete 6502 retention experiment C2 sampling ledger contains 942 roles from 80 S1 worlds. Its 42 eligible roles represent 41 tapes from 11 worlds; role-level world counts overlap. f, Predicted and measured free byte positions agree for all eligible roles. Exact-byte, fixed-core and ancestry checks each have zero mismatches over 91,428 direct cases and 731,424 serial records. Free positions admit arbitrary alternatives jointly inherited with causal roots. Pointer randomization is unmeasured and was not an 8086 S2 arm.

**Sources:**
- `results/census6502/table_main.csv`
- `results/census8086/table_main.csv`
- `manuscript/figures/out/v8/data/law_6502_controls.csv`
- `results/s6502/s1_worlds.csv`
- `results/evolved6502/eligibility.csv`
- `manuscript/figures/out/v8/data/evolved6502_capacity.csv`
- `results/literal_channel/direct_search.csv`
- `results/literal_channel/finite_search.csv`
- `results/literal_channel/CHECK.md`
- `results/s6502/CORRECTION.md`
- `results/census8086/reset_correction_counts.csv`

## Extended Data 7. BFF literal and environmental controls

[PDF](ed8_7.pdf) · [PNG](ed8_7.png)

**Legend (221 words).** a, All archived BFF variants with modal detection, population detection and later collapse counts. Modal detection requires headroom-normalized generation-two parent-byte-match gain of at least 0.3; population detection requires at least half of sampled cells to satisfy the same heritability criterion. Collapse is terminal heritable share below 10% after population detection. Literal writing emits operands; wrap permits pointer wrap; harmless variants prevent unmatched brackets from halting. b, Complete paired fetch-fraction and conditional-entropy counts for first-modal tapes, one per detected world. All distinct observed pairs have explicit tape counts; values are rounded to five decimals. H is offspring entropy conditional on the fixed parent, measuring partner inflow over 256 uniform partners per tape. The standard harmless-bracket variant has no archived entropy panel. c, Literal-no-wrap repetition controls: open fetch can coexist with zero sampled partner inflow. d, Wrap-plus-literal halting controls retain all worlds in counts and final P-tape-share medians at 16,384 BFF population updates, called epochs in the archive. e, Paired first-versus-final world medians of entropy and fetch. f, Registered individuality results pair each pass fraction with its support and rejection thresholds. Z80 claims use 80 Stage G worlds, except the 67 loop-bearing finals; BFF Q4b pools three literal variants of 12 worlds each. Q4b is rejected and Q4a lies between decision boundaries. Entropy, confinement and dependence on initial partner contents remain separate measurements.

**Sources:**
- `results/bff/runs.csv`
- `results/bff_stdnh/runs.csv`
- `results/biology/individuality/per_replicator.csv`
- `results/bff_dials/d2_bandwidth.csv`
- `results/bff_dials/d1_lethality.csv`
- `results/biology/individuality/predictions.csv`
- `results/bff_dials/FINDINGS.md`

## Extended Data 8. Founder genealogies

[PDF](ed8_8.pdf) · [PNG](ed8_8.png)

**Legend (244 words).** a, Z80 L16 discovery and N6 confirmation each comprise 20 worlds. Sixty-four ancestral lines are sampled from each world with confined founders: 17 discovery and 20 N6 worlds. The genealogy classifier requires at least 75% tape-byte copying in at least 32/64 uniform partners; its open/confined classes count any partner opcode or operand fetch in at least/fewer than 32/64 encounters. Open-first refers to the earliest copier on each line. b, First founders and later open-ancestry-to-confined-founder events have separate denominators. c, N6 founding-event categories distinguish rewrites from partial overwrites; recorded mechanisms and registered source classes are separate. d, One executor per N6 first founder is classified under whole-tape and executed-byte copying scopes. A copier reproduces at least 75% of scoped bytes in at least half of partner encounters. All 20 executors are whole-tape non-copiers, but only 11 are executed-byte non-copiers, below 12 required. The probability of all 20 being whole-tape non-copiers under independent draws from pre-founder soup composition is 0.062. e, Chain records and elapsed updates for all 20 first founders. f, Every registered N6 component includes its decision boundary and observation. N6-3 measures cells carrying two-byte words at founder-executed instruction starts divided by all soup cells; its median over 99 word/founder records is 0.00190 against a 0.00024 random-tape baseline, a 7.9-fold ratio. The 55 recorded point-mutation events on 20 pre-founder chains imply 0.0064 expected confined founders; zero of 20 founders are observed mutation completions. N6-4 remains not met; the five-of-20 point-mutation kill is not triggered.

**Sources:**
- `results/ladder/lod_groups.csv`
- `results/lod/founders2.csv`
- `results/lod_n6/founders2.csv`
- `results/lod_n6/founding.csv`
- `results/lod_n6/chain_v6b.csv`
- `results/lod_n6/N6_SCORE.md`
- `results/lod_n6/parts_v6.csv`
- `results/lod_n6/REACH.md`
- `results/lod_n6/REACH.csv`
- `results/lod_n6/PARTS_FIRST.md`
- `results/lod/OPEN_FIRST.md`
- `results/lod_n6/OPEN_FIRST.md`

## Extended Data 9. Registered S2 and separate follow-ups

[PDF](ed8_9.pdf) · [PNG](ed8_9.png)

**Legend (243 words).** a, All 20 registered coherent-fetch 8086 L16 S2 worlds: 14 first-copier detections and six right-censored nondetections at 3,000,000 updates, with open and confined fetch labels. b, Descriptive R2 persistence in the same worlds: zero open memory-fed tapes at all 11 terminal samples; earlier scheduled samples combined contain one ever-positive world. c–d, Registered law accounting from these S2 worlds retains culture-negative, excluded and unresolved candidate tapes. No case is certified eligible: L1/L2 remain untested and L2 support is suspended. e, Separate exploratory admissions and selected post hoc exact-law assays from the same 20 worlds: 779 culture-positive world/tape records lead to 600 admissions, comprising 590 d16 and ten d4 tapes, and then 32 selected genotypes. All 2,224,640 encounters check predicted bytes, core and ancestry without mismatch. Copy displacement d is the write-minus-read byte offset. Nonempty perturbations comprise d4 erasure and d16 single/joint payload; d4 joint payload and d16 erasure have zero applicable coordinates and no corresponding perturbations. Empty-assignment controls remain in encounter counts. f, Exploratory geometry for ten d4 genotypes with hexadecimal byte pattern 5F A4 u v, where u/v are variable bytes; repeated read/write pairs coincide. The separate post hoc snapshot-fate result pools five samples per world, soup/uniform partners and donor A/B roles; medians are totals per 512-encounter batch of non-donor 16-byte halves newly matching the donor’s two-byte tiling, across 20 S2 and 20 Z80 G L16 worlds. S2 uses three donor tiles and Z80 one. These follow-ups do not supply registered evolved-law evidence.

**Sources:**
- `results/s8086/tables/worlds.csv`
- `results/s8086/tables/persistence.csv`
- `results/s8086/tables/law_denominators.csv`
- `results/s8086/tables/law_candidates.csv`
- `results/s8086/exploratory/eligibility.csv`
- `results/s8086/exploratory/law_assay_summary.csv`
- `results/s8086/exploratory/first_family.csv`
- `results/s8086/exploratory/encounter_fates.csv`
- `results/s8086/exploratory/first_family_copies.csv`
