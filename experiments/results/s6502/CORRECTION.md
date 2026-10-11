# Correction to S1 (2026-10-11)

The registration governs. Its heritability definition says gen2 ≥ 0.3 over 64 partners; the cited frozen DZ final-cell estimator uses 32 shared uniform-byte partners. Both final-cell analyses are reported. Its trace text says 16 ‘soup partners’, while the cited exec_trace/DZ implementation generates seeded uniform-byte partners. Random partners remain the reported registered trace analysis, as instructed; soup partners are explicitly a sensitivity analysis.

The corrected 6502 loop detector counts a 16-bit PC decrease only when PC_next differs from (PC + executed instruction length) mod 2^16. Each prefix replay supplies its current memory, so instruction lengths come from the executed opcode even after self-modification. The 256 lengths are exported from Zilion m6502.table(), commit 183ff5c, and hashed in each score. This corrects multi-byte sequential increment wraps such as $FFFE → $0001. Forward ring-alias returns still do not satisfy the registered PC-decrease rule; the prior exploratory position-space finding remains exploratory.

Top-tape cultures retain 64 partners, trace draws retain 16 uniform-byte partners, judge RNG [seed, 0, 11], and final RNG [seed, 23]. Final classes are computed independently at n=32 and n=64 with the same 64 sampled cells, trace draw stream, threshold and frozen estimator. The full heritability selection, up-to-16-cell selection and mutant scans are repeated; this is not restricted to the original scanned cells. Increasing n expands the assay’s generation-one draws, so generation-two draws are not a nested prefix of the n=32 assay. Scan draws may change if heritable membership changes.

Sensitivity uses each corrected random-analysis first tape, without changing t_rep or re-selecting tapes. It chooses the earliest saved snapshot at or after t_rep from 20,000, 100,000, 1M, 3M and 10M; 16 cells are drawn without replacement using independent RNG [seed, 0, 6502, 16]. Partner indices, bytes and snapshot hashes are saved in each world JSON. Only the final snapshot and that required snapshot are read from the volume on Modal. These snapshots can be later than t_rep: this is a snapshot sensitivity, not an exact contemporary-soup reconstruction. No soup-partner P2 history or revised t_closed is inferred.

## Original and corrected results

| arm | protocol | with_rep | first_confined | first_open | first_loop | first_confined_loop | first_open_straight | p2_worlds | km_t_rep | transmitter | regenerator | offset_eligible | offset_equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | original32 | 20 | 20 | 0 | 14 | 14 | 0 | 0 | 1125000.0 | 94 | 209 | 0 | 0 |
| A | v2_frozen32 | 20 | 20 | 0 | 14 | 14 | 0 | 0 | 1125000.0 | 94 | 209 | 0 | 0 |
| A | v2_literal64 | 20 | 20 | 0 | 14 | 14 | 0 | 0 | 1125000.0 | 94 | 209 | 0 | 0 |
| B | original32 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| B | v2_frozen32 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| B | v2_literal64 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| C | original32 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| C | v2_frozen32 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| C | v2_literal64 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| D | original32 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |
| D | v2_frozen32 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |
| D | v2_literal64 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |

| component | original_frozen32 | v2_frozen32 | v2_literal64 |
|---|---|---|---|
| P1 | not supported | not supported | not supported |
| P2 | supported | supported | supported |
| P3 | not tested | not tested | not tested |
| P3_classes | supported | supported | supported |
| P3_offset | not tested | not tested | not tested |
| P4 | supported | supported | supported |
| null | does not apply | does not apply | does not apply |
| kill | not triggered | not triggered | not triggered |

Arm A P1: original 14/20 (0.70); corrected 14/20 (0.70), against ≥ 0.8. All 20 arm-A first tapes remain confined under random tracing. P2 counts are worlds with a qualifying tape strictly before closure. P3’s overall verdict is not tested when its offset component has zero eligible cells; its class-occurrence component is reported separately. B–D are descriptive, with no kill rule.

Across all 80 worlds, changes after the loop fix: {'first_loop': 0, 'first_confined': 0, 'first_open': 0, 't_rep': 0, 't_closed': 0, 'p2_violation': 0}. 80/80 complete world rows and 80/80 frozen32 scanned-cell lists are identical to the original; 285/285 cached top-tape judgments are identical. Exact raw-input hashes and machine provenance match the original outputs. Original result files and REVISION_PREREG.md are unchanged. All worlds retain their full 10M horizon. The independent arm-A KM median in audit_km.csv is 1,125,000 steps.

## Final-cell cultures: 32 frozen versus 64 literal

| arm | sampled_cells | heritable_32 | heritable_64 | heritability_changes | scan_membership_changes |
|---|---|---|---|---|---|
| A | 1280 | 1156 | 1156 | 0 | 0 |
| B | 1280 | 363 | 363 | 0 | 0 |
| C | 1280 | 1174 | 1174 | 0 | 0 |
| D | 1280 | 1186 | 1186 | 0 | 0 |

Arm-A P3 class occurrence: n=32 94 transmitters and 209 regenerators (supported); n=64 94 and 209 (supported). Offset equality: n=32 0/0; n=64 0/0. P3 overall: not tested / not tested. Counts are scanned heritable confined cells; unscanned proportions use the frozen population estimator.

## Trace-partner sensitivity

| arm | first_replicators | random_confined | soup_confined | changed_confined | random_open | soup_open | changed_open | random_loop | soup_loop | changed_loop | changed_any |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 14 | 14 | 0 | 0 |
| B | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 4 | 4 | 0 | 0 |
| C | 20 | 19 | 19 | 0 | 1 | 1 | 0 | 15 | 15 | 0 | 0 |
| D | 18 | 18 | 18 | 0 | 0 | 0 | 0 | 17 | 17 | 0 | 0 |

Across 62 first replicators, 0 change confinement, 0 change openness, and 0 change loop calls; 0 change any flag. Censored worlds have no first-replicator trace and are retained as missing in partner_sensitivity.csv.

## Mutation race

The inherited packed-u32 mutation kernel has the same shared, previously disclosed nondeterminism as the Z80 runs ([ZSOUP.md](../../algocell_exp/ZSOUP.md), lines 203/268, mutation/claim nondeterminism). [audit_mutation_race.csv](audit_mutation_race.csv) supplies a concrete witness under ordinary L16 parameters: batch seed 12, mutations 174 and 180 target distinct bytes 91616 and 91617 in word 22904. Intended values are $3F and $CB; observed values are $3F and $00, losing mutation 180. The original Z80 sim_square_L16.wgsl:1139 also uses non-atomic read/modify/write of the packed word. The nominal rate is 512 byte-replacement attempts per step, not 512 guaranteed applied updates. Re-scoring fixed snapshots does not fix that race or quantify its outcome effect. A mutation-policy correction requires a separately labelled world rerun and does not replace S1.

## Every independent-audit finding

| finding | audit_status | evidence | correction_action |
|---|---|---|---|
| Initial A, X, Y, P, PC | PASS | Generated `zil_reset`/`zil_init_state`, manifests and budget-zero empirical records agree on 0, 0, 0, 0x20, 0. See `audit_cpu.csv`, `reset`. | Retained audit finding; no correction required. |
| First push at 2L−1 | PASS | S=0xFF at registered L16/L32; first PHA writes zero only to position 31/63 and decrements S to 0xFE. Extra L20 probe uses S=0xDF and writes position 39. | Retained audit finding; no correction required. |
| Zero page, zp,X wrap | PASS | Hand memory matches `zp_alias` and `zpX_8bit_wrap` at P32/40/64. `(0xFF+2)&0xFF=1`. P40 distinguishes 8-bit wrap from incorrectly mapping unwrapped 257. | Retained audit finding; no correction required. |
| Absolute addresses beyond P | PASS | `LDA #$5A; STA $1234; LDA $1234; STA $1E` writes the expected `0x1234 mod P` position and position 30 at all three lengths. | Retained audit finding; no correction required. |
| Stack page 0x0100 OR S | PASS | `LDX #0; TXS; LDA #$5A; PHA; LDA #1; PLA; STA $1E` writes position `256 mod P`, pulls $5A, restores S=0. P40 distinguishes page 1 from address S alone. | Retained audit finding; no correction required. |
| BRK vector and seam | PASS | Vector reads use `65534 mod P`, `65535 mod P`, after pushes. At P32/64 a first BRK overwrites its vector with return PC=2. At P40 the vector is positions 14/15, not the ring tail; the planted $5678 vector is followed. | Retained audit finding; no correction required. |
| Branch aliases beyond P | PASS | BNE +56 reaches 16-bit PC=58 and executes ring position `58 mod P`; BNE −3 reaches $FFFF and an immediate operand fetch wraps to address 0. Full final memory and maps agree. | Retained audit finding; no correction required. |
| World dimensions and Z80 parameter parity | PASS | 160×125, 8192 drawn pairs, square lattice, noise_exp=4. Identical `Soup._params` bytes and 22 verbatim soup declarations per arm; only CPU execution is replaced. | Retained audit finding; no correction required. |
| Nominal mutation count/rate | PASS | The inherited parameter is `8192/2^4=512` byte-replacement attempts per step. At L16 this is 512/320000; at L32, 512/640000. It is per pair, not a constant per-byte rate across lengths. | Retained audit finding; no correction required. |
| Applied mutation updates | FAIL | With ordinary L16 params, batch seed 12, mutations 174 and 180 target distinct bytes 91616/91617 in packed word 22904. Expected $3F/$CB; observed $3F/$00. See `audit_mutation_race.csv`. | Disclosed shared Z80/6502 lost-update race; no world rerun or mutation-policy change. |
| Pair packing, inactive pairs, writeback | PASS | A then B, little-endian words, no padding at L16/L32. Foreground bridge probes preserve inactive $77 sentinel memories, give them 0 steps/writes, and reproduce every active B write in the soup. Active cells are disjoint. | Retained audit finding; no correction required. |
| Instruction budgets, especially C | PASS | Bridge INX/JMP loops record 128 steps and X=64 in A/B/D; **512 steps and X=0 in C**. Prefix machine runs also distinguish 128 from 512. | Retained audit finding; no correction required. |
| BRK halting only in D | PASS | Only D's selected manifest has `lethal.firstUnit=0`. Opcode $00 halts before BRK effects, with host reason 2, PC=0 and no writes. A zero immediate operand does not halt. Faithful BRK executes in A/B/C. | Retained audit finding; no correction required. |
| Runner sample schedule | PASS | Independently enumerated 864 samples: 50…2000/50, 2250…20000/250, 22500…1M/2500, 1.025M…10M/25000. Four selected raw worlds match every step and finish at 10M. | Retained audit finding; no correction required. |
| Top-three tapes/shares | PASS | Forced equal GPU hashes do not merge distinct tapes. Full-byte Counter counts from A/65002's final raw soup reproduce all top-three tapes/shares and the byte histogram. No padding participates. | Retained audit finding; no correction required. |
| Seeds and dependence | PASS | All 80 score JSONs/CSV rows cover exactly 65001–65020 once per arm. Same seeds in every arm are explicitly registered. A/C/D share initial soups and deterministic batch-seed streams, so cross-arm results are paired; this is not an unregistered seed reuse. | Retained audit finding; no correction required. |
| Snapshot steps | PASS | Runner code and the four fetched raw records agree on 20000, 100000, 1M, 3M, 10M. A/65002's final snapshot checksum and 320000-byte shape match. | Retained audit finding; no correction required. |
| Culture/trace parity with frozen Z80 DZ | PASS | Both implement 64 uniform-byte culture partners per judged top tape, 16 uniform-byte traces, 64 final cells with 32 culture partners, up to 16 scans with 16 alternatives/site and 16 partners. Seeds/cache order are retained. | 32-partner branch retained; updated scorer re-runs DZ p=0 exactly. Literal64 and soup traces are separate outputs. |
| Registered trace partner source | FAIL | `REVISION_PREREG.md:1169` says “soup partners”; `score6502.py:57` and `population_classes.py:47` generate uniform random bytes. This affects confinement, openness, loops, t_closed, and P1–P3 interpretation. | Both conflicting definitions reported. Seeded random partners retained as reported registered analysis; first-replicator soup-snapshot sensitivity computed for all arms. |
| Registered final-cell heritability count | FAIL | `REVISION_PREREG.md:1168` defines heritable over 64 partners. `population_classes.py:45` uses 32 to select final heritable cells and scan membership. Frozen-Z80 parity and the literal S1 specification conflict here. | Full final-cell classification repeated with both 32 frozen and 64 literal culture partners, including selection and scans. |
| Hand-checked seed 65002 copying | PASS | `b0 38 ca b5 aa 95 02 38` repeated twice copies all B by instruction 89. Six independent hand-oracle prefixes match memory/registers. A separate 64-partner two-generation culture scores gen2=1. | Retained audit finding; no correction required. |
| Confined/open from fetch map | PASS | Seed 65002 fetches only A. A boundary LDA-immediate probe starts at A[15], fetches B[0] as its operand, and marks B in fetchMap but not startMap. Scorer correctly uses fetchMap. | Retained audit finding; no correction required. |
| t_rep and sample traversal | PASS | Replaying raw A/65002 in chronological/share order reproduces all four cached judgments and t_rep=t_closed=70000. All 80 world CSV endpoint fields agree with their score JSONs. | Retained audit finding; no correction required. |
| Arm-A KM | PASS | Exact-rational product-limit calculation gives survival 11/20 at 915000, 1/2 at 1125000, 9/20 at 1150000. Twenty events, no censoring. See `audit_km.csv`; the median is the first time survival reaches 1/2. | Retained audit finding; no correction required. |
| P=0x20 versus core reset; D and I | PASS | `zilion/src/cores/m6502/index.ts` also defines world reset P=0x20. D=0 and I=0 initially. BRK sets I, preserves D. No external IRQ/NMI exists. This is the registered world reset, not a hardware reset sequence. | Retained audit finding; no correction required. |
| JAM in traces | PASS | JAM fetch/start are recorded once, steps stop at 1, trap reason/cause are 3/1, PC parks at the opcode. Budget 128 does not create later transitions. | Retained audit finding; no correction required. |
| Operand fetches; start-map ring wrap | PASS | Fetch records opcode and operand bytes, excluding data reads. JMP $FFFE then a 3-byte NOP starts at ring 30/14/62; its high operand fetch wraps to ring 0. Maps match the hand sets at P32/40/64. | Retained audit finding; no correction required. |
| Registered PC-decrease implementation | PASS | `pc_has_loop` implements the literal frozen rule. Independent regrouping of 41280 exploratory prefix rows yields 14 PC-decrease seeds and six higher-alias-return seeds, agreeing with the registered calls. | Original literal exception reproduced; v2 generalizes the sequential increment exception as instructed. |
| Loop measure as a control-flow-loop detector | FAIL | A straight sequential 3-byte NOP at PC=$FFFE ends at PC=1 and is called a loop: the exception covers only $FFFF→0. Conversely the six forward-alias copiers return in ring position without a 16-bit PC decrease. These are defects of the registered measure. | Corrected sequential increment wraps; forward alias returns remain outside the registered PC-decrease measure. |
| Exploratory artifacts | PASS | 320 saved initial pairs rerun on the vendored core reproduce their final memory, PC and fetch/start maps. All six no-PC-decrease seeds return to earlier A positions in all 16 saved encounters. | Retained audit finding; no correction required. |
| Frozen scorer provenance | PASS | All 80 score JSON source hashes match scorer commit `2f3582c49f548977b516b7199d06eb641a3ed7e1`. Current `dial_soups.py` has later changes, but S1's imported `hx` and `_km_median` ASTs are unchanged. See `audit_sources.csv`. | Original files preserved; every v2 score records source hashes and committed scorer provenance. |
| Full raw-data audit of all worlds | UNSURE | Only four specific raw JSONLs and one final soup were fetched, not all 80 histories/400 soups. Their input/runner/machine hashes and metadata pass; all 80 scored artifacts were checked. | All 80 raw histories and final soups now read and integrity-checked for scoring on Modal; required sensitivity soups also checked. This does not independently replay all full world histories or inspect all 400 soups. |
| Verdict sensitivity to protocol conflicts/race | UNSURE | All 320 saved scanned arm-A cells remain heritable at n=64, with unchanged offsets; max gen2 difference is 0.0110677. Unscanned cells, new scan membership, soup-partner judgments and mutation-race effects were not reconstructed. | 64-partner final classes and nearest-at-or-after first-replicator soup traces quantified; mutation-race verdict effects remain unmeasured. |

## Validation, execution and cost

42 scorer tests and 4 focused executor regression tests passed, including the audit’s sequential 3-byte wrap, genuine backward near-wrap branch, self-modified opcode and existing machine-loop tests. All 256 descriptors match the archived exploratory descriptor table. Z80 DZ p=0 was re-run on Modal L4: 10/10 exact on every column (control/SCORER_CONTROL.md and control_matches.csv). Every new result used its vendored machine, with no shader fallback; Modal saved-result round trips matched every returned result. All 80 results record input, source, core/world/WGSL hashes and scorer commit.

| app_id | worlds | gpu | cpu | memory_gib | max_simultaneous_containers | created_at | stopped_at | app_elapsed_s | scorer_wall_s | scorer_compute_estimate_usd | whole_app_compute_upper_bound_usd | state | tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ap-xVead2ED3QKiTs4BMnXRjp | 80 | L4 | 1 | 2 | 20 | 2026-10-11 09:19:04-05:00 | 2026-10-11 09:21:49-05:00 | 165.0 | 366.98 | 0.08800384277777779 | 0.7913583333333334 | stopped | 0 |
| ap-X86zTQDutPZL4QMGTXKcd3 | 10 | L4 | 1 | 2 | 10 | 2026-10-11 09:18:22-05:00 | 2026-10-11 09:18:55-05:00 | 33.0 | 42.442 | 0.01017782738888889 | 0.07913583333333334 | stopped | 0 |

Rate-priced scorer compute estimate: $0.0982. Conservative whole-app compute upper bound: $0.8705; adding a $0.01 allowance for the small JSON transfers and one month of added score/control-input volume storage gives $0.8805, below the $5 cap. This bound charges each app’s entire created-to-stopped interval at its maximum simultaneous L4 container count, 1 CPU and 2 GiB per container, including startup and idle time; it is not an invoice. Current rates are saved in modal_rates.json, and cost_summary.json records the bound. Both task apps are stopped with zero tasks; the running tier-2 app ap-GpoWKGnLbLpyPF0kyUWKsT was untouched.

Reproduce: run the committed Modal wrapper with --registered --arms A,B,C,D --out experiments/results/s6502/v2; it refuses existing volume score paths. Gather existing results with --gather-only. Then run PYTHONPATH=experiments /Users/neo/repos/algocell/experiments/.venv/bin/python experiments/correction_s1.py. Tables are written only under v2; these two correction texts are generated beside the original audit. See v2/PROVENANCE.json for source and evidence hashes.
