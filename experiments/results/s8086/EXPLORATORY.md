# EXPLORATORY / post hoc — 8086 S2 follow-ups

Every analysis, admission, assay and conclusion here is **EXPLORATORY and post hoc**. The registered [S2 record](S2.md) is unchanged: R0 pass; R1 fail and route kill triggered; L1 untested; L2 untested with unresolved support / incomplete extraction. These follow-ups do not revise those verdicts. No population world was run, no Modal compute was used (**$0 incremental Modal spend**), and nothing was pushed.

**(a) Why did `B9 51` never become a first replicator?**

The exact cause is **not identifiable from these archives**. The best-supported explanation is a bottleneck in birth or establishment, with strong encounter-role effects once the soup has changed. The response depends on entry phase and role. In the `B9 51` entry phase, mature 8086 soup improves active copying: at 20k, planted `B9 51` copies into soup partners at 75.62%, versus 53.20% into uniform partners. When planted as B, however, it survives only 2.36% against soup, versus 25.66% against uniform A. In the opposite `51 B9` entry phase, active copying instead drops to 32.71% against soup versus 54.28% against uniform partners. Thus tar can impair this pusher in a phase-specific way. At step zero the two contexts are closely matched. The role asymmetry can impede establishment; this is an inference, not a measured population growth rate.

No full `B9 51` or `51 B9` tile, and no alternating run of four or more bytes, appears in any of the 11,700 scheduled top-3 samples or 2,000,000 checkpoint cells. Isolated two-byte matches do occur and are tabulated separately. The absence of a four-byte seed in these observations supports an arrival bottleneck, but does not reveal unobserved transient seeds or how they were lost. A four-byte exact motif is rare even in uniform data, and shorter functional precursors could grow between checkpoints; these observations do not estimate a birth rate. Confidence is high in the scan and encounter measurements, moderate that receiving-role vulnerability matters in sampled mature soup, and low in a specific historical cause of the missing first replicator.

The scan covers all 20 canonical complete worlds, 585 scheduled samples per world, and checkpoints 0, 20k, 100k, 1M and 3M; snapshot bytes match their logged hashes. A partial tile means a **linear** alternating run of ≥4 bytes inside a cell; ≥8 and isolated pairs are also recorded. Both physical tile phases are searched. The following counts pool repeated observations, not unique lineages. Words are the literal/memory-fed exact-zero census witnesses across L16/L32 and budgets 128/512; both orientations are searched. L32 is only a census reference.

| EXPLORATORY word / opposite phase | Full checkpoint cells | Checkpoint cells with run ≥4 | Scheduled samples with a full tile | Earliest full observation, steps | Earliest run ≥4, steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b951` / `51b9` | 0 | 0 | 0 | — | — |
| `55bd` / `bd55` | 0 | 2 | 2 | 600 | 600 |
| `57bf` / `bf57` | 0 | 15 | 0 | — | 20000 |
| `07a5` / `a507` | 244 | 6,990 | 41 | 1400 | 150 |
| `ab58` / `58ab` | 0 | 0 | 0 | — | — |
| `5ea5` / `a55e` | 0 | 2 | 0 | — | 1000000 |
| `96a5` / `a596` | 4 | 15 | 0 | 1000000 | 100000 |

`55 BD` first appears as a full top-3 tile at 600 in world 8086019, with only 15 cells (18 at 650); it is below the registered 100-cell share threshold. `07 A5` is observed as full and partial tilings, and the re-extracted culture panel contains many nonperiodic `07 A5 …` full-tape copiers. Appearance of a motif is not a registered detection or a causal classification. Exact occurrences, times, tapes, cell indices and multiplicities are in [motif hits](exploratory/motif_hits.csv.gz); zeros and complete denominators are in [scan CSV](exploratory/motif_scan.csv.gz). [Composition](exploratory/snapshot_composition.csv) records byte fractions and diversity, without defining tar by a single byte.

For the encounter panel, 512 distinct cells are sampled without replacement per world/checkpoint; matched seeded uniform partners are used for every tile. Both orientations use physical entry at A, budget 128. “A copy” below accepts either of the two exact tile phases in B; “B survival” requires the original planted B bytes at their physical phase. Actual inputs and outputs are retained in [raw fates](exploratory/encounter_fates_raw.csv.gz), and [summaries](exploratory/encounter_fates.csv) separate exact-phase copies, pre-existing copies, new copies, donor survival, similarity and B fetch. Each table entry pools 10,240 encounters; cells within a world are not independent evolved worlds.

| EXPLORATORY tile | Checkpoint | A copy: soup / uniform | B survival: soup / uniform |
| --- | ---: | ---: | ---: |
| 8086 `b951` | 0 | 54.11% / 54.14% | 25.12% / 25.52% |
| 8086 `b951` | 20,000 | 75.62% / 53.20% | 2.36% / 25.66% |
| 8086 `b951` | 100,000 | 76.92% / 54.05% | 2.61% / 26.10% |
| 8086 `b951` | 1,000,000 | 74.84% / 53.46% | 2.53% / 24.58% |
| 8086 `b951` | 3,000,000 | 74.41% / 54.04% | 2.44% / 25.08% |
| 8086 `51b9` | 0 | 54.12% / 54.61% | 12.05% / 12.32% |
| 8086 `51b9` | 20,000 | 32.71% / 54.28% | 1.36% / 12.51% |
| 8086 `51b9` | 100,000 | 33.70% / 54.70% | 1.41% / 12.32% |
| 8086 `51b9` | 1,000,000 | 37.71% / 54.91% | 1.53% / 12.24% |
| 8086 `51b9` | 3,000,000 | 39.06% / 55.59% | 1.49% / 12.01% |
| 8086 `07a5` | 0 | 100.00% / 100.00% | 9.62% / 9.74% |
| 8086 `07a5` | 20,000 | 100.00% / 100.00% | 1.54% / 9.48% |
| 8086 `07a5` | 100,000 | 100.00% / 100.00% | 1.69% / 9.43% |
| 8086 `07a5` | 1,000,000 | 100.00% / 100.00% | 1.87% / 9.40% |
| 8086 `07a5` | 3,000,000 | 100.00% / 100.00% | 1.69% / 9.16% |
| Z80 `01c5` | 500 | 76.24% / 57.05% | 56.96% / 27.64% |
| Z80 `01c5` | 1,000 | 71.93% / 56.43% | 56.17% / 26.38% |
| Z80 `01c5` | 2,000 | 64.47% / 56.58% | 53.80% / 27.43% |
| Z80 `01c5` | 20,000 | 56.29% / 56.76% | 33.85% / 26.97% |
| Z80 `01c5` | 100,000 | 75.78% / 56.20% | 6.54% / 27.58% |

The Z80 `01 C5` comparison uses the actual Stage G L16/budget128 executor and all 20 seeds 2001–2020, with local snapshots at 500, 1000, 2000, 20k and 100k. At 500, A-copy success is 76.24% versus 57.05%, and B survival is 56.96% versus 27.64%. Early Stage G soup is thus friendlier to a receiving pusher in this panel. Some Stage G partners already carry `01 C5`; the summary explicitly records these and new-copy counts. Stage G had already undergone pusher expansion by 500, so ecology and expansion are confounded. S2 has no dense pre-20k population checkpoints; different times, machine semantics, mutations and population histories preclude an architectural or causal establishment claim. Late Z80 soup also suppresses B survival. The comparison supports context dependence, not a universal 8086/Z80 ordering.

**(b) What is `5F A4 x y`, and does it carry heritable variation?**

Yes: all ten observed first-family genotypes (10 of 14 registered first copiers) transmit multiple exact alternative genotypes through eight fresh-partner passages. It is a four-byte forward-overlap copier: `POP DI; MOVSB; branch`. Initial SP=1 makes POP read A[1]=A4 and A[2]=x, so DI=0xXXA4 and DI mod 32=4; SI=0, ES=DS=0, SP becomes 3 and DF=0. Each MOVSB reads DS:SI and writes ES:DI, then increments SI and DI. All ten exemplars make 64 byte copies in 128 steps, with displacement 4, and write every A and B coordinate. The first moves are 0→4, 1→5, 2→6, 3→7; overlap propagates the first four ancestral bytes throughout the ring. B starts receiving copies at SI=12 / DI mod32=16. Final source map is `[0,1,2,3]` repeated four times, R=K=C_R={0,1,2,3}, E={4,…,15} and F is empty under the conservative law footprint.

The two variable bytes supply a taken control transfer. The frozen NMOS core interprets 60–6F as conditional-jump aliases, so ordinary modern disassembly alone would be misleading. Conditions stay true because POP/MOVSB preserve the relevant reset flags; LOOP wraps initial CX=0 to 65535, staying taken for the encounter. For rel8 branches, `4 + signed(y)` is 1 modulo 32, returning to the physical MOVSB byte. Raw IP can grow, wrap or repeat while its mapped position remains in A. The `E9 3C` exemplar fetches a third displacement byte at physical A[4], already overwritten with 5F; its displacement is 0x5F3C, and next IP 5 + displacement again maps to A[1]. Its raw IP advances by 24,384 each cycle. This explains confinement without requiring repeated *raw* instruction-start positions; the original descriptive one-pass/loop labels are unchanged.

The following table details the ten registered first-family exemplars. The all-world panel below also covers later and low-share family observations. Full step-by-step read/write/fetch/start events, first source/destination moves, numeric DI and physical fetch coverage are committed in [family traces](exploratory/first_family_traces.csv.gz) and [exemplar summary](exploratory/first_family.csv). The [per-copy pointer CSV](exploratory/first_family_copies.csv) reconstructs every full 16-bit SI/DI state from reset, the single POP and DF=0 MOVSB increments, and checks every mapped read/write against the trace. It includes each raw IP, source ancestor, byte value and before/after pointers; reconstruction is explicitly labelled.

| EXPLORATORY seed | x y | Frozen decode | DI after POP | Physical fetch positions | Transmitted alternatives at positions 0 / 1 / x / y |
| --- | --- | --- | ---: | --- | --- |
| 8086001 | `63 3d` | JAE rel8 (NMOS 60–6F alias) | 25508 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |
| 8086005 | `73 5d` | JAE rel8 | 29604 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |
| 8086006 | `6d 7d` | JGE rel8 (NMOS 60–6F alias) | 28068 | [0, 1, 2, 3] | 1 / 0 / 19 / 7 |
| 8086009 | `61 3d` | JNO rel8 (NMOS 60–6F alias) | 24996 | [0, 1, 2, 3] | 1 / 0 / 19 / 7 |
| 8086012 | `e2 7d` | LOOP rel8, initial CX=0 wraps to 65535 | 58020 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |
| 8086013 | `7d fd` | JGE rel8 | 32164 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |
| 8086014 | `e9 3c` | JMP rel16, displacement formed by y and overwritten/fetched byte 4 | 59812 | [0, 1, 2, 3, 4] | 1 / 0 / 0 / 7 |
| 8086017 | `61 bd` | JNO rel8 (NMOS 60–6F alias) | 24996 | [0, 1, 2, 3] | 1 / 0 / 19 / 7 |
| 8086018 | `69 dd` | JNS rel8 (NMOS 60–6F alias) | 27044 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |
| 8086020 | `63 fd` | JAE rel8 (NMOS 60–6F alias) | 25508 | [0, 1, 2, 3] | 0 / 0 / 19 / 7 |

The descriptive X1 scan varies each of the four founder coordinates through all 256 values, repeats the mutant founder four times, challenges it against 16 fresh uniform partners, and separately performs eight serial fresh-partner encounters. There are 10 genotypes, 10,240 assignments (10,200 non-baseline alternatives), and 245 alternatives that pass all 16 exact physical-phase challenges and all eight serial passages. Each genotype has at least seven transmissible y alternatives; nine also have 19 x alternatives. Four have one transmissible byte-0 alternative. Results establish constrained **heritable variation in the executed founder**, not arbitrary payload freedom. Mutations here alter every homologous founder byte; the law erasure assay below instead mutates a single physical coordinate. [All X1 assignments](exploratory/family_transmission.csv) and [serial hashes](exploratory/family_serial.csv.gz) retain failures as well as successes. This small finite panel supplies no establishment or fitness claim.

A separate **all-world descriptive trace panel** covers every world with a scheduled exact `5F A4 x y` tiling: 18 of 20 worlds. All 18 selected exemplars copy at the exact physical phase against all 16 fresh census partners, with the four-byte geometry. Seven were **unjudged** by registered culture because their maximum scheduled counts stayed below the 100-cell detection threshold; they are not classified culture-negative. The additional world 8086003 has culture-positive later family members after its different registered first copier. Worlds 8086008 and 8086015 have no scheduled full-family tape, and no such full-family checkpoint cell. Across all five checkpoints, only world 8086003 contains full-family cells (1,061 repeated observations / 21 genotypes); the scheduled record retains many transient families missed by coarse checkpoints.

| EXPLORATORY seed | Scheduled family genotypes | Registered culture-positive among them | Selected x y | Registered selected status | Exact / 16 fresh partners | Selected max scheduled cells |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 8086001 | 2 | 1 | `63 3d` | culture-positive | 16 | 263 |
| 8086002 | 1 | 0 | `eb 1d` | unjudged | 16 | 34 |
| 8086003 | 25 | 18 | `e2 bd` | culture-positive | 16 | 409 |
| 8086004 | 2 | 0 | `71 3d` | unjudged | 16 | 75 |
| 8086005 | 4 | 3 | `73 5d` | culture-positive | 16 | 248 |
| 8086006 | 13 | 6 | `6d 7d` | culture-positive | 16 | 462 |
| 8086007 | 3 | 0 | `75 5d` | unjudged | 16 | 86 |
| 8086008 | 0 | 0 | — | absent | — | 0 |
| 8086009 | 7 | 5 | `61 3d` | culture-positive | 16 | 115 |
| 8086010 | 2 | 0 | `73 fd` | unjudged | 16 | 55 |
| 8086011 | 1 | 0 | `e2 7d` | unjudged | 16 | 43 |
| 8086012 | 3 | 2 | `e2 7d` | culture-positive | 16 | 120 |
| 8086013 | 3 | 2 | `7d fd` | culture-positive | 16 | 642 |
| 8086014 | 4 | 1 | `e9 3c` | culture-positive | 16 | 274 |
| 8086015 | 0 | 0 | — | absent | — | 0 |
| 8086016 | 2 | 0 | `79 7d` | unjudged | 16 | 52 |
| 8086017 | 3 | 2 | `61 bd` | culture-positive | 16 | 108 |
| 8086018 | 7 | 2 | `69 dd` | culture-positive | 16 | 195 |
| 8086019 | 1 | 0 | `65 9d` | unjudged | 16 | 87 |
| 8086020 | 3 | 3 | `63 fd` | culture-positive | 16 | 116 |

For this panel the storage observer records **actual full SI and DI before every instruction**, rather than reconstructing them. All 306 full lane memories and original machine-record words matched uninstrumented execution. [Coverage](exploratory/family_world_coverage.csv), [exemplars](exploratory/family_world_exemplars.csv), [all 17-context pointer/fetch/data traces](exploratory/family_world_traces.csv.gz), and [per-MOVSB actual pointers/read-write pairs](exploratory/family_world_copies.csv.gz) preserve the result. Event kind 5 stores SI in the position field and DI in the value field; [the observer source](exploratory/family_pointer_observer.wgsl) is archived. The SHA in the table identifies compiled source after removing its leading explanatory label comment. This additional descriptive panel is outside the 779-candidate eligibility denominator and outside the 32-genotype exact-law intervention panel. It does not revise a registered detection.

**(c) What changes when POP ES and the other original missing opcodes are modelled?**

The separate [exploratory verifier](../../exploratory8086_law.py) transfers full-word dependencies for POP ES/CS/SS/DS and register POP, stack/segment address dependencies, frozen 60–6F aliases, ALU taint, IN's frozen all-ones return, LES/LDS, far JMP, byte immediates and register exchange. Segment taint is deliberately conservative despite the tiny address ring. Fifteen [unit/integration tests](../../tests/test_exploratory8086.py) check numeric registers and ordered stack reads against the frozen machine, verify complete full-tape and four-byte source maps, and reject arithmetic memory writes as copies. Registered `law8086.py`, machine WGSL and S2 verdict files were not edited.

All 779 culture-positive genotype-world cases were re-extracted. Admission requires agreement across all 17 archived ordered contexts; an excluded or unresolved case stops at its first failed certificate. Post hoc admission is 600 eligible across 20 worlds, 72 excluded and 107 unresolved; L1 remains zero. Subsequent unsupported dependencies still block admission, so modelling the first missing opcode does not certify every original unresolved case.

| EXPLORATORY original registered reason | Cases | Now eligible | Now excluded | Still unresolved |
| --- | ---: | ---: | ---: | ---: |
| original_partner_read | 55 | 0 | 55 | 0 |
| unmodelled_dependency_opcode:07 | 708 | 587 | 16 | 105 |
| unmodelled_dependency_opcode:0c | 2 | 1 | 0 | 1 |
| unmodelled_dependency_opcode:0f | 1 | 0 | 0 | 1 |
| unmodelled_dependency_opcode:21 | 1 | 0 | 1 | 0 |
| unmodelled_dependency_opcode:31 | 1 | 1 | 0 | 0 |
| unmodelled_dependency_opcode:5f | 10 | 10 | 0 | 0 |
| unmodelled_dependency_opcode:e5 | 1 | 1 | 0 | 0 |

The newly admitted geometries are **590 full-tape copiers with d=L=16**, identity source map, R=all A and E=empty; and **10 four-byte overlap copiers with d=4**, F=empty and E={4,…,15}. The d=16 cases have forward contiguous byte moves but do **not** reduce a 16-byte genotype to a smaller overlap founder: they carry arbitrary unexecuted full-tape coordinates according to their held code footprints. All observed admissions have source start 0 and physical phase 0. No smaller-seed claim is assigned to those full-tape cases. [Every certificate and exclusion](exploratory/eligibility.csv) and [reason totals](exploratory/eligibility_counts.csv) are retained.

Before assaying, [the 32-genotype panel and all K/R/C_R/F/E/source-map predictions](exploratory/law_panel_predictions.csv) were committed in `5bbc9d7a2e321bb45d0384975f9b3b4abb555ac1`. Selection took the smallest held footprint in each of 20 worlds, all ten first-family genotypes, and two deterministic additional cases. Thus there are 22 full-tape and ten four-byte cases across 20 independent evolved worlds; it is a purposive capped panel, not a random estimate for all admissions.

The **exact frozen assay function** was reused in independent globals, with the extended mismatch verifier, logged foreground GPU wrapper, and 64 independent seeded joint assignments (`joint:256`…`joint:319`) replacing its 512-joint panel. It checks every physical output byte with the predicted source map, all 17 original census partners, all-value single substitutions, all-value mutable erasures, and serial passages to g=8 for every single/joint assignment. No rotation or favorable alignment is introduced; no candidate is removed after a mismatch.

Observed: **0 candidate mismatches / incomplete cases in 2,224,640 encounters**, 66,048 single assignments over 258 genotype-coordinate pairs, 2,048 joint assignments, and 30,720 erasure assignments. All 32 selected genotypes completed with exact matches. Serial streams, generation-zero source maps, physical phase and actual input/output/partner hashes appear in [per-genotype encounter CSVs](exploratory/law_encounters/); [summary](exploratory/law_assay_summary.csv) and [denominators](exploratory/law_assay_denominators.csv) distinguish challenge and generation. For d=4, F is empty: its 64 joint assignments repeat the baseline and offer no payload evidence; the erasure checks are substantive. For d=16, E is empty: erasure controls are inapplicable. There are zero held-erased coordinates in the selected panel. An independent [CPU audit](exploratory/verification.csv) recomputed every expected offspring hash, census/serial input hash, serial partner byte stream, source map and per-case denominator against the frozen assignment generators. Finite assays support these post hoc predictions only; they do not turn the registered L2 result into a pass, certify the remaining unresolved candidates, or establish unrestricted robustness for every eligible genotype.

**(d) Is publication provenance archived?**

Yes. 239 files from the S2 worktree launch directories and billing-poller are byte-preservingly archived under [launch/exploratory-publication-archive](launch/exploratory-publication-archive/), including nested scoring/recovery runs, ledgers, watchdog logs, handoffs, invocation scripts, tests and resource readings. Files over 1 MB were deterministically gzipped, without rewriting historical contents. [The archive manifest](exploratory/publication_archive.csv) records every original path, archived path, both hashes and byte counts. Stored files total 12,646,770 bytes; the largest is 1,039,122 bytes, below 50 MB. The historical launch records remain historical provenance; their presence is not new compute authorization or a changed registered interpretation.

**Reproduction and resource record (EXPLORATORY / post hoc).** Downloaded data were read with `/Users/neo/repos/algocell/experiments/.venv/bin/python -I`. Local canonical S2 downloads and local Stage G copies supplied all inputs; no new download or remote compute was needed. Executor/verifier byte hashes are in [source identities](exploratory/compute_source_hashes.csv). Input hashes are in [S2 inputs](exploratory/input_hashes.csv) and [Stage G inputs](exploratory/stageG_input_hashes.csv). All 4731 local GPU operations were synchronous foreground calls with equal measured pauses; maximum burst 1.302s, total call time 30.043s, total pause 40.648s. [Burst ledger](exploratory/gpu_bursts.csv) includes compilation and tests. All committed files are below 50 MB.

Run from the repository root; scripts add only the trusted local experiments directory to isolated Python's import path:

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py scan
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py reextract
# Commit predictions before the law stage; execute GPU stages sequentially.
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py encounters
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py transmission
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py world_family
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py law
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py archive
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086_verify.py
# Commit the CSVs, then regenerate this presentation without compute:
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py report
```

[Analysis source](../../exploratory8086.py), [GPU assay wrapper](../../exploratory8086_compute.py), [CSV-to-Markdown generator](../../exploratory8086_report.py) and [CPU hash audit](../../exploratory8086_verify.py) are committed. Re-running local assays appends to the burst ledger; timings and Git prediction identities are provenance, not byte-stable scientific outputs. Registered report generation is deliberately separate.
