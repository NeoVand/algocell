# Paper study for the Algocell app refactor

Read-only study of `/Users/neo/repos/algocell-paper`, 2026-10-13. Paths below are relative to that repository unless labelled app paths. This document is research input to `APP_PLAN.md`, not a new registration or a claim that the browser replays archived stochastic worlds.

## Sources and authority

Read `manuscript/MAIN_v9.md`, the complete `manuscript/v9_inputs/STORY_TALK.md`, the design contract and main/supplementary plans in `manuscript/FIGURE_PLAN_v9.md`, and `figures/soup_stills.py`. Read the relevant original registration sections in `registrations/originals/PLAN.md` (system, A/C/F/G/H/I), `REVISION_PREREG.md` (R3 resets, E/E-L32L competitions, DZ dial, S1, S2 and later amendments), and `BIOLOGY_PREREG.md` (assembly/individuality). Inspected `registrations/chronology.csv`, `registrations/freezes/INDEX.csv`, `provenance/snapshot.json`, machine world JSONs, condition-generation code, assay code, and the generated result tables/reports referenced below. Also read app `CLAUDE.md`.

Authority order matters: MAIN_v9's careful scientific wording supersedes the more expansive lecture; final generated/corrected reports supersede old exploratory prose; dated registration amendments supersede earlier text. In particular, the S2 registration and world JSON retain old prelaunch notes, but `results/s8086/S2.md` records the completed study. Original registration text is historical evidence and must not be mistaken for the current outcome. The paper snapshot binds source commit `6052bcc42ffc83c61c35d12c51afd6260414879e`; S2 has its own nested freeze and world identities.

The paper currently identifies Zilion 0.3.0 and frozen machine exports. A new app running 0.3.1 must say which contract it implements and record its own identities; it cannot silently label a new world as an exact archived paper run. Historical `compat: '0.2'` is a scientific identity, not a mere performance setting.

## Scientific story and public language

The central question is: **what does a reproducing program preserve?** The paper connects four distinct phenomena: random code accumulates orderly but non-reproducing products; some programs reproduce while retaining few tested variants; copying geometry determines which differences survive within a specific family; environments change the prevalence of those designs. The public opening can be “Copying is not inheriting” with the printing-press example, but avoid the lecture's absolute claim that a particular replicator “could never evolve.” The paper retains working core variants and rare transmission.

There is no fitness score, culling rule, or externally imposed task. Programs spread by executing writes into partners that may themselves reproduce. The lattice, encounter schedule, reset state, memory boundaries and budgets are nevertheless supplied by the world. This is an inspectable digital model, not a reconstruction of Earth's origin of life or proof of biological autonomy.

Do not present an inevitable staircase from tar to pusher to confined regenerator to transmitter. The benign Z80 history is conditional on that world; longer tapes and other processors differ. The 8086 specifically falsified the predicted open beginning despite an available pusher. Likewise, a ret-containing exemplar and a pusher are independent examples unless linked by recorded genealogy.

The four measurements must remain separate in the interface:

1. **Culture:** does a tape produce offspring which in turn reproduce against fresh partners? A two-generation functional assay, not visual regularity.
2. **Confinement:** does instruction-stream fetching enter the partner? Fetch includes operands and prefixes, not just instruction starts. It is not partner independence, no-data-read, or a heredity classification.
3. **Inflow:** fix the parent, vary partners, and measure output variation. Zero observed output entropy for a finite panel is not universal independence.
4. **Transfer:** introduce a parental difference and track it separately from reproductive survival. Equality of byte values does not establish causal ancestry; markers can be reacquired from partners.

The app's first charts should explicitly be descriptive: zero-byte fraction, exact top-tape share, and exact-genotype diversity. None measures life, fitness, R/T functional class, or evolutionary progress. A transmitting population can consist of many distinct payload-bearing tapes, so its largest genotype can be rare even while transmitters dominate. Show chart denominators and whether sampling or a complete census was used.

## Shared paper world contract

- Square lattice 160×125, 20,000 tapes. Four cardinal neighbours with **reflecting boundaries**, not a torus. App hex/mixed worlds are exploratory alternatives, not this contract.
- 8,192 proposed pair slots per update. Collision claims mean fewer active encounters; display proposals and realized active pairs separately if available.
- A then B, each L bytes; standard ring P=2L. Execution begins at physical A byte 0. Writes to both halves persist.
- Default budget 128 execution steps per encounter. Z80 block-repeat and 8086 REP iterations consume individual steps; this is not 128 complete block moves.
- Background `noise_exp=4` means `8192 / 2^4 = 512` uniformly located byte-replacement **attempts** per update. Values uniform 0…255, so replacements can leave values unchanged. Packed-u32 concurrent mutation writes can lose updates. Do not claim 512 applied changes. Attempt rate is 0.0256 per cell/update and 0.0016 per byte/update at L16.
- Random initialization is seeded SplitMix64 words, little endian. GPU claim order and packed mutation races prevent exact population trajectory reproducibility from a seed. Export complete checkpoints plus core/world/manifests and scheduling settings.
- Historical Z80 resets all registers/flags to zero except SP; `topAliases: 'lastByte'` makes SP the largest 16-bit value mapping to B's final byte. Z80 decrements before storing, so the first pushed byte lands at `2L−2`.
- 6502 uses A=X=Y=0, P=0x20, PC=0 and `firstPushAt: 'lastByte'`. First push lands at `2L−1`; NMOS JAM halts. Faithful BRK pushes PC+2 and P then follows the vector, with architectural wrapping before ring mapping.
- 8086 uses AX=CX=DX=BX=BP=SI=DI=0, CS=DS=ES=SS=0, IP=0, FLAGS=0xF002 (DF=TF=0), coherent fetch, hidden latch `zero_ea`, and `firstPushAt: 'lastByte'`. SP=1 on P32/64; the first pushed word has its low byte at P−1 and high byte at 0. No extra lethal-zero rule. Coherent instruction semantics do not emulate a prefetch queue or hardware bus timing.

Sources: `src/algocell_exp/machines/worlds.json`, `worlds_6502.json`, `worlds_8086.json`; `src/make_conds.py`; `registrations/originals/PLAN.md:15`; `REVISION_PREREG.md` S1/S2 machine contract; `results/s6502/CORRECTION.md`; MAIN_v9 Methods.

## Exact preset catalogue

Each preset should expose: source/section, evidence status, full machine/world contract, population settings, seed and initial condition, explanation, what to look for, and the qualified historical finding. A smaller browser population is an explicit **interactive scale** derivative with different statistics; never silently substitute it for the source's 20,000 tapes. Start paused. Set an automatic ≤60-second running burst for local review.

### 1. Historical Z80 — Stage G, L16

`isa=z80, compat=0.2`, P32, budget128, k4, 160×125, 8192 slots, benign zero, no ablation, zero registers with historical SP. Random start, seeds 2001…2020 (default 2001); horizon300,000. Source: PLAN Stage G (line283), `src/make_conds.py::stage_g`, `results/stageG/FINDINGS.md`, `results/stageG/stageG/stage_g_runs.csv`.

Explanation: watch zero-rich structure form, then test whether abundant tapes actually reproduce. The common early `01 c5` pusher writes its own instruction bytes into the partner and executes into partner addresses. What to look for: falling diversity and rising top-tape share may precede or accompany replication; select a tape for culture and fetch-map tests. Finding: all 80 historical Stage G worlds (four lengths) had open first detected copiers; at L16, 19/20 first exemplars were `01 c5` tilings, with median detection450 updates. That is a cohort observation, not a guarantee for this browser run. Later modal-tape observations do not substitute for random-cell class estimates.

### 2. Benign versus lethal zero — Stage I

Benign reference is Stage G above. Lethal arm changes only opcode-first-byte rule: `halting.lethal.firstUnit=0`, probability1, L16/P32, compat0.2, budget128, k4, seeds4001…4010, horizon300,000. Sources: PLAN Stage I lines793 onward, `results/stageI/per_world.csv`, MAIN_v9 environment section, `src/algocell_exp/machines/worlds.json` `z80c02_lethal_L16`.

Explanation: zero is an ordinary harmless NOP in benign Z80; in this intervention a zero at an instruction start halts the encounter before execution. A zero used as data or an operand is not automatically lethal. What to look for: compare zero abundance, first culture-positive tape and observed partner entry under each chemistry. Finding: in the historical lethal cohort all ten first copiers were confined with zero sampled partner inflow and at most four scored sites; nine of ten final populations had transmitters as a majority of culture-positive cells. This is not proof that zeros alone cause selection for increased capacity; ecological mechanism is unresolved.

### 3. Lethality dial — DZ

Same L16 Stage G contract with fresh random halt draw **per zero-opcode fetch**; p∈{0,0.01,0.03,0.1,0.3,1}, ten worlds per p, seeds9001…9010, horizon300,000. Historical compat0.2 and separately labelled corrected compat0.3 are both meaningful. Sources: REVISION_PREREG DZ lines534 onward, `results/dial/DIAL.md`, `results/dial/dial_worlds.csv`, corrected `results/tier2/T2_DZ.csv`, `T2_COMPONENTS.csv`.

Explanation: the dial changes risk each time zero is fetched as the first instruction byte. What to look for: compare p0, p0.1 and p1, preserving world size/reset/budget; a short run illustrates mechanism but cannot estimate emergence-time distributions. Finding: historical open-first counts were 10,10,10,10,2,0 across the six probabilities. Historical closure-timing and endpoint clauses did not all pass. Corrected-core open-first counts were 10,10,10,9,4,0; transmitter-majority counts0,1,1,3,3,8. Do not mix corrected and historical verdicts or promise a monotonic transition time.

### 4. Random registers — R3 C

Z80 compat0.2, L16/P32, budget128, k4, standard lattice, benign chemistry, horizon300,000, seeds8001…8020. Every encounter independently randomizes A,F,B,C,D,E,H,L, all alternates, IX,IY uniformly; keep PC0 and historical SP. Random-SP is a separate arm (SP uniform16-bit, other registers zero), not the same checkbox. Sources: REVISION_PREREG R3 C lines334–348; `src/algocell_exp/gen_conv_shader.py`; `results/conv/REPORT.md` and `conv_worlds.csv`.

Explanation: many compact copiers exploit zeros supplied by the reset. What to look for: the load–push pusher initializes its own data and can still reproduce; test the same four-byte block copier under zero and random resets. Finding: load–push was first in20/20 random-register worlds; no population closed in the initial300k cohort, but later3M experiments did produce confined designs, so “impossible” is false. The browser RNG must be recorded; its stream need not match the archived bespoke shader draw.

### 5–6. Ring geometry — L12 on24 versus28

Z80 compat0.2, L12, P24 control or P28 intervention, budget128, k4, no ablation, seeds101…110, horizon300,000. P28 has A12+B12+4 zero padding; padding is reset for each encounter and never written back. SP must alias **B's last byte23**, not padded ring's final byte27. Sources: PLAN Stage F/F4 lines240–282; `results/stageF/stage_f/rings.csv`, `NUMBERS_F.md`; `src/make_conds.py::stage_f`.

Explanation: changing discarded padding changes address aliasing without changing tape size. What to look for: isolated `01 c5`×6 cultures and population takeovers differ across rings. Finding: P24 had6/10 detections, KM median245,500; P28 had10/10, median350, all period2 first tapes; P36 also10/10 median600. This is a delayed/altered arrival regime, not an absolute “no-life boundary.” Some other registered arithmetic predictions failed; not every visible period divides P.

### 7. No block copy

Historical Stage C arm: compat0.2, L16/P32, budget128, k4, seeds101…110, horizon300,000. Suppress `family:block-copy` only (LDI,LDD,LDIR,LDDR; explicit ED bytes A0,A8,B0,B8). Sources: PLAN A/C; `src/make_conds.py::ABLATIONS`, `results/stageC/stage_c/NUMBERS_C.md`; app `src/lib/presets.ts` existing preset.

Explanation: remove the dedicated memory-copy machinery while leaving load–push and other routes available. What to look for: pusher and exchange/stack-based designs can still spread. Finding:10/10 detections under128/k4, median350 in this cohort versus200 in its unablated comparator. This does not establish copying impossible without LDIR.

### 8. Strict no-copy (historical preset name)

Historical Stage C1: compat0.2,L16/P32,budget128,k4,seeds101…110,horizon1,000,000 (also32/k2 arm). Suppress exactly `family:ld8`, `ld8-imm`, `ld8-mem`, `ld16-imm`, `ld16-mem`, `ld-special`, `stack`, `ex`, `block-copy`. **CALL/RST and read-modify-write writes remain.** Sources: PLAN Stage A/C; `src/make_conds.py::ABLATIONS['no-copy']`; app current preset.

Explanation: this is a strict load/stack/exchange ablation, not a proof that memory cannot be written. What to look for: regular zero-rich or other structure can persist while cultures remain negative. Finding:no detections in20 no-copy worlds across the two C1 settings by1M; rmw-only similarly none. These finite absences are not universal impossibility. Label “No-copy ablation” with exact removed families, avoiding “all memory writes disabled.” Suppressed Z80 instructions consume opcode/prefix bytes only; their operand bytes become subsequent instructions. Preserve the archived length contract in disassembly and execution.

### 9. NMOS6502 — S1 arm A

`isa=m6502, variant=nmos, compat=0.3`; L16/P32,budget128,k4,standard lattice, faithful BRK,JAM as-chip, registers/stack as above; seeds65001…65020,horizon10M. Source:`worlds_6502.json` `m6502_pair_maps_L16`; REVISION_PREREG S1 lines1133 onward; corrected `results/s6502/CORRECTION.md`, `v2/arm_comparison.csv`.

Explanation: the same encounter ecology with a different instruction vocabulary and explicit stack convention. What to look for: use the6502 disassembler; 00 is BRK, not Z80 NOP. Emergence can take substantially longer than a quick browser tour. Finding:20/20 primary first copiers were confined; only14/20 met the registered additional loop requirement, so that prediction was unsupported. KM median1,125,000 updates. The separate retention study's42 eligible roles/41 tapes/11 worlds supported conditional copying predictions; it does not rescore the failed S1 criterion.

### 10. Coherent-fetch8086 — S2 arm A

`isa=i8086, variant=nmos, compat=0.3, params={prefetch:'coherent', hidden_latch:'zero_ea'}`; L16/P32,budget128,k4,standard lattice, SP1/reset above, no extra lethal rule. Seeds8086001…8086020,horizon3M. Source:`worlds_8086.json`; REVISION_PREREG S2 plus option(b) amendment; `results/s8086/S2.md` and `tables/worlds.csv`. L32 **was withdrawn prospectively**; it remains valid fixed-input context, not a completed registered population arm.

Explanation: current memory is visible to subsequent instruction fetches, with retained REP decode. What to look for: instruction lengths, stack phase and segment translation; compare an available `b9 51` pusher with actual evolved four-byte family exemplars. Finding:14/20 detected;2/14 first copiers open and12 confined, triggering the registered kill. Both registered evolved-law families admitted zero eligible cases and remain untested. Later32-genotype tests are post hoc support, not rescue of that empty assay.

### 11. Offset-switch competition — seeded R/T

Standard Z80 ecology,budget128,k4, L32/P64; mutation-on300k; start50:50 random mixture. R=`04 5e ed b0`×8; T=`a0 5e ed b0`+28 independently random bytes **per T cell**. Benign and lethal conditions. Historical compat0.2 and corrected compat0.3 must be separately named. Other registered arms: mutation-off100k;1% invasion each direction300k. L16 uses R=`44 5e ed b0`×4 and T=`b0 5e ed b0`+12 independently random bytes. Source:REVISION_PREREG E/E-L32L; `src/offset_switch.py`; `results/offset/offset_switch.csv`; corrected `results/tier2/scored/E/REPORT.md`.

Seed bookkeeping: condition seeds1…10 (mixtures),1…5 (invasions); actual historical soup seed=`7000+100*(L==32)+50*(lethal)+seed+1000*startIndex+20000*(mutationOff)`, with startIndex0=mixture,1=R_into_T,2=T_into_R; NumPy initial-design generator seeded `[soup_seed,99]`. New JS-generated planting must disclose stream differences.

Explanation: R copies a short prefix repeatedly; T copies the full tape, distributing its descendants across distinct tails. What to look for: functional frozen **motif-class** shares T/(R+T), alongside absolute core prevalence and diversity; top exact tape share is inadequate. Finding:corrected L32 median T share0.05 benign versus0.76 lethal;10/10 lethal endpoints above0.5. Reciprocal invasions ended near environment-specific compositions, but finite runs do not prove equilibrium. R/T differ in payload background as well as displacement; this does not isolate selection on capacity alone.

### 12. Displacement-edit demonstration

Use compat0.2,L32/P64,zero-reset,budget128,mutation off for isolated assay. Baseline=`04 5e ed b0`×8. Edit first byte04→60(hex), so96 mod64=32; this changes displacement4 to full-tape32 while keeping opcode semantics viable. `a0` is another valid full-distance setting used in competition. Do **not** change04→20(hex) blindly: the first byte is also an instruction. For marker demonstration, introduce a non-core byte and compare its actual descendants at d4 and d32. Source:STORY_TALK§10; `results/capacity/parents.csv`,`edit_eligibility.csv`,`CAPACITY.md`; MAIN_v9 copying theorem.

Explanation: forward writes eventually read earlier writes; overlap tiles the initial prefix. What to look for: marker disappearance outside the protected prefix at d4, preservation at d32 under the qualifying contract, and retained edited machinery. Finding:conditional free payload r=d−c; c4 here gives0 versus28. It requires d dividing L, coverage, intact machinery, noninterference and reset viability. It says nothing about adaptive use of payload, all possible core variants, or every copier. Registered101 settings from22 parents yielded1,067,608 agreeing direct cases; not that this simple browser panel reproduces that full assay.

### 13–14. Pusher and return-closer examples

compat0.2,L16/P32,benign zero,zero-reset,budget128. Pusher=`01 c5`×8. Return closer=`ad e3 21 e3 21 c0 ad c0`×2. Plant an explicitly labelled demonstration (single cell, fraction, or whole culture), or inspect isolated pairs; do not label a seeded population spontaneous emergence. Sources:REVISION_PREREG R3 B; `results/stageG/stageG/stage_g_runs.csv`; `results/branching/branching_summary.csv`.

Explanation: both can reproduce, but their execution routes and marker fates differ. What to look for: pusher partner fetching versus return-closer confinement; culture versus transfer results. Finding:the registered branching protocol retained48/64 unmutated pusher populations through8 generations, with no one of512 mutation types meeting carriage criterion. The return closer retained all64 unmutated controls but scored no transmissible site. Do not translate no scored site into literal absence of every marker copy or every possible hereditary alternative.

Additional8086 exemplar for the inspector: seed8086001 first tape `5f a4 63 3d`×4, from `results/s8086/tables/worlds.csv`; a post hoc d4 family, not a generic REP copier. `07 a5`×8 is a separate census construction with robust exact copying and open fetch; open fetch does not imply surviving original-partner information.

## Culture implementation specification

Port the operational definition rather than equating one exact copy with reproduction. Source:`src/algocell_exp/assay.py`, MAIN_v9 Methods.

For selected tape T and each fresh uniform partner R: execute T as A, retain resulting B as offspring O. Compute best cyclic-shift matching fraction S(T,R) and S(T,O); normalized gain=(after−before)/(1−before), informative partners exclude those already≥0.75 similar. For generation2, execute every first-generation O against fresh independent R2, including unsuccessful first-generation outputs; compare to original T using the same normalized gain. Report average two-generation gain, first-generation≥75% copy fraction, sample count, budget, ISA, world/ablation and seed. Paper culture-positive threshold is gain≥0.3, with n64 usual for top tapes. All initially saturated/noninformative panels produce unknown/NaN, not zero/sterile. The app may offer16-partner quick screen but must label it a smaller screen;64-partner mode is suitable for matching the paper's nominal sample size.

Fetch maps classify **sampled encounters**: open=partner entry in≥8/16; confined=none in16; intermediate=1…7; unknown=not assayed. An overlay of a single live encounter must say “partner entered this encounter” or “no observed partner entry,” not label a stable biological class. Unassayed cells remain unknown. Instruction starts and all instruction-stream fetches need separate styling.

For marker transfer use an explicit mutant/wild-type control, fresh partners per generation, offspring gate and generation count. For causal provenance, equal byte values are insufficient. Initial simple marker UI should accurately call results retention of a marked sequence under this test; full ancestry/source propagation is a later instrument feature unless implemented. The registered branching protocol is2 partners/member, retain offspring only, cap32 uniformly,8 generations,≥75% best-shift survival, centered5-byte markers absent from wild type, majority carriage. Do not present a simpler serial test as this protocol.

## Design system extracted from the paper

- White canvas; charcoal `#25313A`; background/unclassified `#E6E8EB`; open `#83909A`; transmitted payload/full-tape T `#168A8D`; erased variation/partial-copy R `#D85A3A`; lethal chemistry `#8653A8` only when encoding chemistry. Never reuse one colour simultaneously for ISA, assay confidence and reproductive class.
- Arial throughout; use tabular numerals for measurements. Web type sizes must be accessible; do not carry tiny print-figure point sizes into UI. Use restrained fine rules, sparse chart grids, direct labels, and generous space. Meaning through redundant labels/shapes/hatching rather than hue only.
- The lattice is the visual anchor. Preserve cell index/spatial position; never sort genotype colors into an attractive but fictitious colony. Use nearest-neighbour rendering without smoothing or contrast transforms that erase rare cells.
- Paper byte palette: `h=(value*0.6180339887)%1`, HSV saturation0.70, brightness0.82 for odd values/0.62 for even; zeroRGB(250,250,250). Draw each tape as a ceil(sqrt(L)) square, padding white; for L16 this is4×4 pixels/blocks. Byte colors encode byte values only, not ancestry or function.
- Class view highlights abundant exact genotypes, with persistent assigned colors and singleton grey; all-zero tape white. This is **not** a functional R/T map. Functional maps require independently assayed cells and unknown state for the rest.
- Controlled motion should communicate execution/write movement or recorded lineage only. Avoid decorative arrows implying a genealogy between illustrative exemplars.

## About draft suitable for adaptation

**Algocell — copying is not inheriting.** Algocell is a live world of short machine-code programs. Neighbouring tapes meet in shared circular memory. A processor executes one tape, both keep the writes, and random byte replacements introduce variation. There is no assigned goal or fitness score. A program spreads only by writing material that can reproduce in another encounter.

The research asks what that reproduction preserves. Order can appear without a self-reproducer. A lineage can keep reproducing while losing an introduced difference. Some copying designs regenerate a short pattern; others transmit variable sequences they never execute. Within a specified family of forward copiers, the distance between reading and writing determines which parental positions remain free to vary. A byte in the copying machinery can set that distance and itself be inherited.

The companion paper, _From random code to heritable variation in digital primordial soups_, studies these distinctions using culture, execution, transfer and recorded-history tests. Its findings concern declared digital worlds. It does not show adaptive use of the transmitted payload, selection for increased capacity, or open-ended evolution. Presets cite the registered conditions; your live run is a new stochastic realization, not a replay of a published result.

Zilion powers the processor and world model. The app provides historical Z80 compatibility0.2 alongside corrected Z80, NMOS6502 and coherent-fetch8086 under the0.3 contract. Processor behavior and world settings have separate identities so that reset values, rings, halting and budgets stay visible. The8086 contract reads current memory after writes; it does not reproduce a hardware prefetch queue. Historical BFF experiments remain in the research archive because BFF is not an exported npm core.

Algocell and Zilion are by Mostafa ‘Neo’ Mohsenvand, MIT Media Lab. This work builds on computational primordial soups by Blaise Agüera y Arcas and colleagues, and the Z80 soups of Cicala and colleagues. Credit the SingleStepTests project for the Z80,6502 and hardware-recorded8088 validation suites, with the exact suite/contract links documented by Zilion. The interface runs locally in your browser; simulation bytes are not sent to a service. Link the app source, Zilion, paper when an owner-approved public URL exists, and reference records. Do not invent a published paper URL or a durable data accession: those remain author-supplied placeholders in MAIN_v9.

## Guide draft structure and accurate instruction text

1. **Begin:** choose a preset, read its explanation and declared settings, press Run, and Pause before inspecting. Run advances population updates; Step advances a declared number of updates. A60-second burst limit makes local work bounded, and the update counter distinguishes computation from elapsed wall time.
2. **Read the lattice:** byte view shows values in stable spatial positions; exact-tape view shows repeated sequences; fetch overlay shows observed partner entry or unassayed status. Click a tape to inspect it. Selection should also work by keyboard and explicit cell index.
3. **Read the inspector:** byte address/hex and ISA-aware instruction decoding; distinguish instruction starts, operands, fetched bytes and untouched data. Reset/budget/halting are part of the interpretation. A disassembly is a static reading, not proof of the path actually executed after self-modification.
4. **Change the world:** explain ISA/version, tape length, ring padding, lattice dimensions/topology, pair slots, budget, reset convention, mutation attempts, seed, lethal byte/probability, and per-ISA ablations. Changes that require a new world reset it visibly; settings that apply on the next encounter say so. Same seed does not promise GPU trajectory replay.
5. **Run a culture:** selected parent→fresh partner→offspring→fresh partner. Report sample count, copy fraction and normalized two-generation gain. Culture positivity is an operational test, not life or adaptive evolution. Distinguish one exact-copy observation from the culture.
6. **Follow a byte:** plant/edit a marker, retain wild-type control, choose transfers, watch marker survival separately from lineage survival. Describe whether this is value matching, sequence carriage or true ancestry; do not imply the third from the first.
7. **Compare experiments:** benign/lethal; d4/full-tape displacement; L12/P24 versusP28; load–push with/without block-copy operations; switch ISA and inspect its different reset/stack behavior. Keep all other parameters fixed. Quick browser runs explain mechanisms, not paper-cohort statistics.
8. **Interpret plots:** zero fraction=zero bytes/total bytes; top-tape share=largest exact genotype count/tape count; diversity=declared exact genotype statistic (unique count or unique fraction, label precisely). Changes reflect composition; they do not by themselves establish replication, heredity or progress.
9. **Presets and planting:** registered random starts versus seeded demonstrations; saving a custom preset saves a contract, not current population. Explain seed distribution and planting fractions; seeded R/T mixtures draw separate payloads per T cell.
10. **Save and restore:** checkpoint stores complete bytes, dimensions, update/RNG state where available, settings and engine/core/world identities; export manifest JSON for citation; CSV for measurements; tape hex for an exemplar. Old checkpoints import as historical Z80 with explicit assumptions and validation, never silently convert their ISA. A changed engine requires a clear compatibility notice or explicit new run.
11. **Limits/glossary:** update vs encounter vs instruction step; observed confinement vs inflow; copy vs culture; exact genotype vs functional class; marker retention vs causal inheritance. BFF and full registered cohort/branching/provenance workflows are separately marked if deferred.

## Risks and owner decisions

- Historical exact encounter compatibility is feasible; bitwise whole-population reproducibility is not promised by the source science. Preserve/disclose packed mutation and activation policy rather than silently repairing them in historical mode.
- Random-register and probabilistic-halt RNG draws in the old bespoke shaders do not automatically equal new per-lane Zilion streams; characterize the world contract and include a new stream identity.
- Padded rings must anchor stack to B's end, not global padded end. This is an easy implementation error with direct scientific impact.
- A short culture can be completed locally; full marker scans and the registered branching tree need separate batching, progress and cancellation. Never colour a whole population from one selected tape's assay.
- App public paper link, exact credits wording, and whether the default is historical Z80 or corrected0.3 remain owner choices. Recommended default: approachable corrected0.3 exploratory world, with the explicitly historical Stage G preset one click away.
- Authoritative paper palettes are separate from byte-value colours. Avoid an accent system that makes lethal violet simultaneously mean selected ISA.
- Retain complete source citations and scientific limits in the plan/help; cards can present shorter versions with disclosure. Accessible UI labels must not say “life detected” or “heredity” based solely on top-tape share.
