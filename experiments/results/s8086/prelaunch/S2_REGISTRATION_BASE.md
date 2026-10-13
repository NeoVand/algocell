

## S2 — the third machine: the coherent-fetch 8086 (registered 2026-10-12, before any S2 world is run; spend cap $60 approved by the user 2026-10-12)

The user approved the design in `planning/PREREG_8086_DRAFT.md` on 2026-10-12, including the $60 hard cap and the separate broader L2 family. The approved numerical design and all exclusions below are binding. Draft-era “proposed” language in the carried-over rationale and historical cost envelope describes the origin of those choices; it does not create reserve arms or permission to alter the design. The measured final cost ledger supersedes the draft's provisional dollar estimate. No S2 population, pilot, short evolving soup or population observation has occurred. The manuscript has not been edited.

**Prelaunch status: FROZEN, STOPPED BEFORE LAUNCH.** The complete measured forecast exceeds $60. In addition, L2 has a known fixed-input specification defect for nonzero source starts and a conservative unresolved-proof scope. Both are disclosed below; neither changes L1 or silently changes the approved design. No launch receipt exists, and the dispatcher refuses this over-cap design.

### 1. Basis: what is established before S2

The authoritative inputs are [CENSUS.md](results/census8086/CENSUS.md), [MECHANISM_CONTRACT.md](results/census8086/MECHANISM_CONTRACT.md), [LAW_CONTRACT.md](results/census8086/LAW_CONTRACT.md), their CSVs and [table_hashes.csv](results/census8086/table_hashes.csv). They replace the preliminary SP=0 inventory; the exploratory 29/13/10 counts are not registration inputs.

All 256 period-one and 65,536 period-two words were evaluated at each L=16/32 and budget=128/512, against zero and 16 frozen random partners. There are no unevaluated words. Period-two constants duplicate period one; both reversed phases are retained. Counts below are nonzero words giving an exact cyclic-phase copy into a zero partner, **not** genotype frequencies, probabilities of establishment or arrival rates.

| L / encounter budget | Exact period-one words | Exact period-two words | Literal writer | Memory-fed one-pass copier | Confined copy loop | Other |
|---|---:|---:|---:|---:|---:|---:|
| 16 / 128 | 0 | 20 | 12 | 2 | 0 | 6 |
| 16 / 512 | 0 | 29 | 12 | 1 | 0 | 16 |
| 32 / 128 | 0 | 26 | 11 | 5 | 0 | 10 |
| 32 / 512 | 0 | 26 | 11 | 4 | 0 | 11 |

Mechanism counts classify the traced **zero-partner episode through its last partner write**. They are not claims that all contexts have that mechanism. Matching an untouched zero partner is not writing.

Three observations determine the proposed route tests:

- `B9 51` is a literal writer with exact zero bootstrap at both proposed lengths. Exact copies occur in 13/16 random partners at L16 and 6/16 at L32, at budget 128. It has original-partner fetches and output dependence; it is not a partner-independent founder.
- `07 A5` is a memory-fed one-pass copier in all 17 audited encounters at both lengths, exact in zero and 16/16 random partners, with full partner coverage, open fetch in 17/17, zero original-partner fetch/data accesses and zero panel output difference. This is a read-only extraction from `words_L16_k2_b128.csv`, `words_L32_k2_b128.csv` and the corresponding `mechanisms_*.csv`, word `07a5`. The second L16 exact memory-fed word, `ab58`, has no exact copies among its 16 random partners. Thus even within a mechanism class, witnesses have different robustness.
- `FF 30` and `AD 50` are **not** exact zero-partner bootstrappers under SP=1. `AD 50` has length/budget-dependent near-copy performance. Neither is promoted to an exact founder because of its opcode name or the old SP=0 results.

The `07 A5` observation motivates risking a distinct open memory-fed occurrence/persistence prediction. It does not establish all-value heredity, universal partner independence, ecological persistence or an arrival time. The census's robust-near-copy column (at least 75% similarity in at least 8/16 partners) is not the culture threshold below.

Constructed byte and word REP controls are already complete: all 340 geometry/partner/budget rows qualified before payload assays, and the frozen control interventions had no mismatches. This is constructed evidence, not an evolved-law result. No controlled pointer-randomization census exists. **Drop that arm entirely from S2**; it is neither an unreported sensitivity nor a reserve arm. A future pointer intervention requires its own controlled census, design and approval before population work.

### 2. Machine and cohort contract

#### Machine

Use only `i8086_pair_maps_L16` and `i8086_pair_maps_L32` from [worlds_8086.json](algocell_exp/machines/worlds_8086.json) and [PROVENANCE_8086.md](algocell_exp/machines/PROVENANCE_8086.md): Zilion `39a60429a9b16c77cd845f6087062c4e2cc12c53`, core 0.3.0, compat 0.3, variant `nmos`, package status `experimental`, `prefetch=coherent`, `hidden_latch=zero_ea`.

| Identity | Frozen SHA-256 |
|---|---|
| Core, both lengths | `cc05989457d50c769c204f6c6213bc5f9da27c5c4637d40556618ba5cdf477af` |
| World L16 | `49dc7897ad949b92a9b4dada09d206c66ef75307e6fa682c7d45d89299c3fcac` |
| WGSL L16 | `4fee1c7c0ce17f0a4ec7d6ad425b0ebe10342336cb55acbe86ebfe663525a694` |
| World L32 | `ba1ecae7ca965ce4a8be3de870643234813060600e2bf5613aa4206fadbc6f97` |
| WGSL L32 | `f031d3a7d217c9eed92d0e080da90399c3fd88ff8c40869d14739f99b98f9966` |

The abstraction is **8086/8088 instruction semantics validated against hardware 8088 vectors, with coherent fetch**. The 3,137,000-case ledger has 3,126,917 raw architectural passes, 10,083 expected divergences under the declared hidden-latch rule and zero unexpected divergences. It does not validate a prefetch queue, 8086 bus timing or hardware ecology. [Zilion I8086.md](../../zilion/docs/I8086.md) supplies the complete contract.

At every encounter reset: AX=CX=DX=BX=BP=SI=DI=0, DS=ES=SS=CS=0, IP=0, FLAGS=`F002`, DF=TF=0. `firstPushAt:lastByte` sets **SP=1** for P=32/64. The first pushed word writes low byte at P−1 and high byte at 0; SP=0 is excluded. These are initial states; instructions may change them. Physical translation and architectural wrapping precede ring mapping. Register LEA/LES/LDS trap; register far transfers use the frozen fresh zero-EA abstraction. Retain the manifest's interrupts, halt, I/O and undefined-form policies; no extra lethal-tar rule.

An encounter budget of 128 means 128 execution steps, including one step for each REP iteration. REP retains decode across continuations/resumes; CX=0 copies nothing. It is not 128 complete REP instructions. Pair memory is A then B on a ring P=2L, with A entering at phase zero.

#### Worlds and records

| Arm | Role | Worlds | L | Encounter steps | Horizon per world | Seeds, inclusive |
|---|---|---:|---:|---:|---:|---|
| A | Primary route and evolved-law cohort | 20 | 16 | 128 | 3,000,000 | 8086001–8086020 |
| B | Secondary scope cohort; scored separately | 20 | 32 | 128 | 3,000,000 | 8086001–8086020 |

Keep the charter's default size and horizon. The final census changes the expected mechanisms but supplies no arrival-rate estimate that justifies changing either. Same-number seeds across lengths are bookkeeping, not matched trajectories or independent paired causal interventions. There are 120,000,000 planned world-steps in total. No L16/512, pointer, flag, stack-phase or other population arm is included.

Use S1's `ZSoup`/shared soup kernels: 160×125 lattice (20,000 cells), four cardinal neighbors with **reflecting boundaries**, 8,192 drawn pair slots per world-step, the existing collision claims and writeback. This is not a torus and not 8,192 guaranteed active encounters. Record realized active-pair totals separately from draws. Initial bytes follow `Soup.initial_soup` (seeded SplitMix64 words, little-endian), with no planted founders.

Retain `noise_exp=4`: **512 uniformly located byte-replacement attempts per world-step**, replacement values uniform over 0…255. The shared packed-u32 mutation kernel can lose concurrent updates, including writes to different bytes in the same word; replacement can also leave a byte unchanged. Do not claim 512 applied updates or 512 changed bytes. The attempted rate is 0.0256/cell/step, or 0.0016/byte at L16 and 0.0008/byte at L32. Applied-update counts are unmeasured unless separately instrumented and validated; they must not be inferred from a soup difference after encounters. Preserve this disclosed S1/Z80 comparison policy, not an unannounced mutation repair. Hardware and claim/mutation races mean a world seed is not a promise of bitwise replay.

Record exact top-three byte sequences, integer counts/shares, concentration statistics, run metadata and input/output hashes at:

- step 0 (initial-state record);
- 50…2,000 every 50;
- 2,250…20,000 every 250;
- 22,500…1,000,000 every 2,500;
- 1,025,000…3,000,000 every 25,000.

There are 584 post-initial samples, plus step 0. Save complete soups at 0, 20,000, 100,000, 1,000,000 and 3,000,000; the step-zero soup is also reconstructible but its hash must be stored. Keep exact tapes distinct despite hash collisions. Sort ties by lexicographic byte sequence, not discovery order. Log actual completed steps, adapter/backend, resource allocations, versions, source/machine hashes and checkpoints. Additional safety checkpoints do not add endpoint samples.

All worlds run to the fixed horizon, with no success/failure-driven early stop. A biological non-arrival in a complete world is right-censored at 3M. Report first detection as the observed sample time, with its preceding-sample interval; do not call it exact birth. A step-zero hit is a baseline detection, separately flagged. `t_closed` remains censored if an open copier persists. A missing/corrupt sample, technical crash or spend stop is an incomplete record, not a 3M negative. No replacement seeds or selective reruns: any necessary restart is logged, preserves the failed attempt and uses a prospectively specified recovery procedure. Confirmatory arm verdicts require its 20 complete, scoreable records; partial-arm summaries remain provisional.

### 3. Shared measurements and route predictions

#### Culture, traces and detection

Use `algocell_exp/assay.py::assay_many` through the explicit 8086 executor, budget 128, no suppression, `zero_halts=False`, with **n=64 fresh uniform-byte partners in each of two generations for every tape**, including final random cells. For similarity s, use best cyclic byte match to the original parent. Generation-two score is the mean `(s_after−s_before)/(1−s_before)` over informative partners with `s_before<0.75`. All first-generation offspring enter generation two, including unsuccessful copies. A detected replicator requires finite `gen2_score≥0.3`. The ≥75% match and ≥0.5 faithful-offspring fraction remain reported fields, not additional heritability gates. NaN is unassessable, not zero; do not redraw until a preferred verdict appears. An unresolved culture result for a top-three tape at share ≥0.005 makes that world's route history unscoreable: it cannot be skipped to declare a later first event or an absence. Attempt only fixed-input technical diagnosis; if unresolved, report the arm incomplete. Unassessable random-cell cultures remain explicit unknowns in the 64-cell denominator and do not enter the law panel.

The primary trace panel is **16 seeded uniform-byte partners**, not soup partners. On the full 128-step encounter, count consumed opcode, operand, displacement and prefix bytes in B. Open means B fetches in ≥8/16 encounters; confined means none in 16/16; 1–7/16 is an intermediate/context-dependent category. REP continuation starts do not invent new instruction fetches. These classifications do not assert causal inflow or allele retention.

Reuse S1's chronological top-three endpoint structure:

- `t_rep`: first scheduled sample, including zero, with a top-three exact tape at share ≥0.005 (at least 100/20,000 cells) passing the culture test. At that sample choose the largest share, then lexicographically smallest tape on a tie. This is the world's first detected replicator.
- `t_closed`: first such sample with a culture-positive tape confined in all 16 traces. This confinement endpoint does not require a loop or a law-eligible core and is not itself proof of causal closure.
- Scan **all** samples for occurrence and persistence, including after both first events. S1's early exit after `t_rep` and `t_closed` must not truncate these endpoints. A top-three detection protocol can miss rare/nonmodal copiers; the random-cell law sample addresses a different question and cannot retroactively move `t_rep`.

Keep the four encounter classes exactly as frozen in `MECHANISM_CONTRACT.md` and `census8086_trace.py::mechanism`: **literal writer; memory-fed one-pass copier; confined copy loop; other**. Use `DiagnosticMachine(events=8192)` as in the census, checking memory and the entire original lane record against uninstrumented execution. Classify the episode ending at its last partner write. Literal values need admitted reset/immediate provenance. Memory-copy values need paired actual same-value reads/writes. Confined copy loops additionally require repeated **full instruction-start addresses** and no B fetch in that episode; admitted one-pass forms have distinct full starts. Mixed/unresolved provenance and trace overflow stay other. No-change RMW stores remain in coverage and access counts, but do not add new-value provenance. Record the reason for every other label. Overflow does not discard the culture result or the uninstrumented full-encounter fetch map; it prevents a resolved mechanism/geometry certificate.

For each tape publish all 16 labels/counts. A tape has a majority class if ≥9/16 labels agree; otherwise report a heterogeneous context profile, not a fifth encounter mechanism. Define the specific **open memory-fed event** used below as a detected tape for which ≥9/16 of the *same encounters* are both memory-fed one-pass by the episode classifier and open by full-encounter fetch. It therefore also meets the ≥8/16 openness criterion. No literal or unresolved write is pooled into this numerator.

For the same 16 pairs, also execute a zero-partner reference and report, independently:

- unique partner positions written and write transactions;
- original-partner instruction-fetch reads and original-partner data reads separately (reads before that address's first write);
- maximum final-B Hamming difference from the zero-partner output across the 16-member panel, including untouched positions;
- full-encounter open fetch versus episode confinement, without equating either to partner dependence.

The first-write access measure is an upper bound/access proxy, not a full causal-inflow count: unchanged stores and partner-derived writes can complicate material ancestry. In exact law assays, follow parental versus partner provenance explicitly. An open copier with zero measured original-partner accesses is a valid outcome; neither openness nor zero accesses decides its payload capacity.

#### Proposed numerical predictions and verdicts

These are deliberately risky empirical thresholds motivated by availability and robustness, **not rates calculated from motif counts**. Arm A is primary. Arm B receives the same predeclared endpoint calculations and thresholds as a separate secondary scope test, with no pooling or rescue of A.

| ID | Endpoint and denominator | Pass | Failure / insufficient information |
|---|---|---|---|
| R0, detection sufficiency | Number D of complete worlds with `t_rep`, out of 20 | D≥10 | D<10: the detection target is not met and R1 is uninformative about route frequency; still report all detection/censoring times. |
| R1, open beginning | O/D, where O counts first detected copiers open in ≥8/16 traces; denominator includes confined, intermediate, literal, memory-fed and other first copiers | With D≥10, O≥ceil(0.8D) | With D≥10 and a smaller numerator, the open-first prediction fails. Intermediate/other cases are not removed from D. |
| R2a, memory-fed occurrence | Worlds with at least one open memory-fed event at any scheduled sample through 3M, out of all 20 | At least 2/20 | Fewer than 2/20 fails this finite-horizon occurrence prediction, including when D<10. |
| R2b, persistent open memory-fed route | Worlds with an open memory-fed event at **every one of the 11 scheduled samples 2.75M, 2.775M, …, 3M**, out of all 20 | At least 1/20 | Zero such worlds fails this finite-horizon persistence prediction. One missed qualifying sample breaks the interval; do not interpolate it. |

R2b allows the qualifying exact genotype to change between samples; it measures persistence of an observed mechanism class, not genealogical continuity. Report a same-genotype-throughout indicator separately. R2a and R2b have separate verdicts; the joint R2 claim is met only if both pass. Their small absolute-world thresholds are chosen because the validated `07 A5` witness is robust across the tested contexts; the census cannot estimate whether even these thresholds will be attained. The stronger “memory-fed majority of founders,” “all open writers are literal” and obligatory pusher→REP succession are not predictions.

Tabulate the complete first-copier mechanism distribution, stratified by open/intermediate/confined, at both tape and encounter levels. Literal and memory-fed routes stay separate even when both contribute to O. No literal-share threshold is inferred from the 12/20 zero-partner motif ratio. A literal or other open copier persisting to 3M is also an observed result, although it alone cannot satisfy the deliberately narrower R2b.

**Route kill rule:** with D≥10 in A, if at least `ceil(0.25D)` first copiers are confined, withdraw the proposed census-motivated open-beginning claim for the 8086. Report R1's failure even when this stronger kill does not fire (for example, excess intermediate cases). R2 failures withdraw their respective occurrence/persistence claims. No route failure kills the evolved-law assay, alters its eligibility, or removes the machine from the paper. A persistent open copier is not a failure to close.

Report arm-specific Kaplan–Meier detection summaries, censoring counts and exact world counts. Binomial intervals are descriptive; the predeclared integer cutoffs, not a fitted p-value or pooled estimate, decide the predictions. There is **no directional 8086-versus-Z80 arrival-time prediction**: motif counts and the partner panel are not establishment rates, and the architectures differ in more than those counts. Historical Z80 timing may be shown as labeled context with its protocol/hardware/compatibility identity, not a prospective S2 rate claim.

### 4. Evolved-law sample, eligibility and exact test

#### Sampling before any retention result

For every complete world in each arm, form a candidate register from: the first detected replicator; the tape establishing `t_closed` (same share/tie rule); all three final top tapes; and **64 cell indices sampled uniformly without replacement from the 20,000-cell 3M snapshot**. Missing first events add no substitute. The cell sample is chosen before its genotypes, culture scores or mechanisms are examined. Keep every sampled cell and its multiplicity; deduplicate exact genotypes only for execution, never for a cell-prevalence denominator. At most 69 source records per world enter this register.

Run the same n=64 culture on every candidate. Record all candidates, finite/nonfinite scores, inclusion source, traces, and exclusion reasons. The confirmatory evolved-copier panel requires the culture criterion plus the independent geometry certificate below; culture-negative candidates are retained in the register, not silently discarded from sampling counts. Include open as well as confined candidates in the extraction attempt. Fetch openness is not an eligibility rule by itself.

Apply the frozen extractor to every culture-positive candidate **before running any payload retention intervention**. After eligibility is archived, assay all eligible distinct first-event/final-top genotypes (at most five per world), then fill to a maximum of 16 eligible distinct genotypes per world from the random-cell set, in the keyed random order specified in §5. Report every eligible but unassayed genotype as excluded by the predeclared cap. Never replace a failing assayed case with the next candidate. The maximum confirmatory panel is 320 genotype-world cases per arm; identical bytes across worlds retain their world memberships even if deterministic computations are cached.

Publish denominators separately: 20 worlds; sampled cells/records; unique candidate genotypes; culture-positive candidates; eligible candidates; selected assay cases; completed cases; coordinates, alternatives, joint assignments, partner challenges and generations. For random-cell prevalence, use all 64 cells per world, including duplicate genotypes, not only selected assays. Report arm-specific coverage and the number of independently evolved worlds represented; construction rows do not enter those denominators.

#### Eligibility scope, fixed independently of retention

For S2's confirmatory exact law, use the **already validated six-byte REP family** in `LAW_CONTRACT.md`: `BF dLo dHi 49 F3 A4` or `BF dLo dHi 49 F3 A5`. Do not fit a general c from observed site scores. Other evolved geometries receive explicit exclusions from L1 and a descriptive dependency trace. The user-approved, separately scored L2 extension is frozen below; it never rescues or replaces L1. This narrow choice risks an empty evolved panel and accepts that outcome.

For each candidate, a code-and-ordered-trace certificate must establish all of the following at the fixed budget 128, against zero plus the 16 frozen census partners:

1. The actual sampled baseline parent is d-periodic, with the stated six-byte core at entry phase zero; `6≤d≤L`, `d|L`, and d is even for MOVSW. Do not rotate, repair, tile a prefix or otherwise construct an eligible parent from an ineligible evolved tape.
2. Initial DS=ES=SS=CS=SI=0, CX=0, DF=TF=0 under the frozen reset. `MOV DI,d; DEC CX` leaves the specified forward displacement and CX=FFFF before REP. The copied core has c=6 because of its code/operand footprint, not because `d−observed_sites=6`.
3. Source/destination byte pairing is same-valued, forward, with constant byte displacement d on the pair ring; word operations have the frozen low/high ordering. Track original source identity, not just coincident byte values or `assay.copy_offset` (a cyclic alignment statistic, not the overlap period).
4. No intervening operation, original-partner read, conflicting write, executed payload or payload-dependent address/stop condition interferes. REP retains the verified decode; it runs for budget−2=126 copying iterations, below count exhaustion. Static code reasoning plus the trace establishes why arbitrary payload values cannot change that operation.
5. Writes cover the entire partner; the baseline offspring preserves all six core bytes at phase zero and resets into the same admissible family. No cyclic realignment or ≥75% whole-tape gate substitutes for core preservation. The same causal argument applies after transfer.

The primary seed is positions 0…d−1. Predict **exactly the free payload coordinates 6…d−1**, hence d−6 coordinates: L16 d8/d16 → 2/10; L32 d8/d16/d32 → 2/10/26. Predict that every initial position d…L−1 is overwritten before contributing its original value to the offspring. Core/displacement bytes must be inherited exactly while held fixed. Core alleles are not free payload positions, and extra heritable core variants do not refute a payload count.

Freeze the extractor **code before any S2 world is run**. After selecting evolved candidates, commit/hash the full eligibility and exclusion ledger, actual geometry traces, source mapping, predicted coordinates and assay selection **before any evolved payload intervention**. That second data freeze applies the existing rule; it cannot invent new rules. Use the same discipline as the constructed `qualify`→frozen ledger→`assay` workflow. Failure to extract a qualifying evolved core is not a law pass or failure.

#### Interventions and expected descendants

At budget 128 only, for every selected eligible evolved parent:

- Test all 256 values at each primary free coordinate separately (255 alternatives plus baseline), holding the core fixed. Change only the primary seed coordinate, not all repeated copies.
- Test 512 joint assignments: indices 0…255 are counterbalanced full assignments, with payload index j=0…d−7 denoting position 6+j and taking `(37*j+v) mod 256` for assignment v=0…255. Indices 256…511 are fixed independent random joint assignments, keyed as specified in §5. This does not exhaust `256^(d−6)` combinations; the all-combinations statement is conditional mathematics, not a claim of exhaustive experimental enumeration.
- Challenge every single and joint case against zero plus the exact frozen 16-partner panel in `results/census8086/partners.csv` (seed `0x8086C0DE`). The expected offspring is the complete mutated primary d-byte seed repeated L/d times, at phase zero. Check every coordinate, including core/displacement.
- Mutate every initial position d…L−1 through all 256 values as erased-position controls, one at a time against the same 17 partners. Expect the unmodified repeated primary seed, and verify loss of that original source identity. For d=L there are no such coordinates: report zero controls, not a failed or successful positive denominator.
- Serially pass every single and joint case through generations 1…8, using the actual previous B output as the next A, reset machine state and a fresh, independently keyed uniform-byte partner each generation. No continuing mutation and no seeded payload in partners. Compare the full offspring to the original expected mutated repeated seed at **every** generation. Follow the complete core haplotype and d. Zero/frozen-panel one-generation assays and independent serial streams are separate challenges.

Use all ten constructed L/d/operation combinations at budget 128 as pipeline positive controls, with their already frozen control assignments and eight-generation expectations. Their archived budget-512 results remain census evidence, not another S2 evolved assay arm. Controls never substitute for an empty evolved sample. Inflow/provenance controls must distinguish accidental equality or partner reacquisition from descent from the mutated parent; ambiguous ancestry is diagnosed, not counted as causal retention.

#### Pass, mismatch and diagnosis

An eligible exact case has zero tolerance: a wrong predicted payload value or location, a causally retained excluded original coordinate, an altered/lost inherited core or displacement, or an incorrect expected offspring at any registered partner/generation is a **candidate mismatch**. Record its complete input, output, reset state, budget, trace and source identities; stop the affected case for diagnosis, preserving earlier rows. Do not average it into “80% agreement,” search for a favorable alignment, or change d/c after observing retention.

Replay a candidate mismatch twice from identical saved inputs with the uninstrumented frozen machine and verify that diagnostic execution matches its full original lane record and memory. A reproducible, provenance-resolved mismatch that remains admissible under the predeclared hypotheses **fails the law's application to that evolved case and the arm's exact-law prediction**. One such case is sufficient; majority agreement cannot rescue it. A theorem error and an application/implementation error are different diagnoses, but neither permits reporting the failed registered application as supported.

The following require diagnosis rather than an immediate mathematical refutation: an uninstrumented/diagnostic discrepancy; corrupt input or wrong machine hash; a trace overflow; an independently demonstrable failure of an already stated eligibility hypothesis; or apparent retention supplied by a partner instead of the parent. Record whether this invalidates a run, an extraction certificate or only an interpretation. A failed mutant cannot simply be reclassified “ineligible because it failed”; exclusions must follow pre-existing conditions evidenced independently of the retention outcome. If the promised noninterference/descendant closure breaks despite the certificate, that is a failed application. Extinction after a demonstrated hypothesis violation is not evidence that the conditional theorem itself is false. Scorer defects suspend the affected verdict and require a disclosed correction, with original output retained.

A ≥75% similarity failure, a sampled mutscan-site disagreement, additional inherited core alleles, or a failed culture threshold in a different context is not by itself an exact payload mismatch. Conversely, similarity/culture success cannot excuse an exact mismatch. Report offspring viability separately and diagnose its loss; it is not a post hoc filter for deleting otherwise eligible failures.

For each arm, **supported on the tested evolved panel** requires at least one eligible evolved case, every selected case and required control completed and valid, and zero reproducible eligible mismatches. Report the independent-world count prominently; one genotype is not proof of broad ecological coverage. With no eligible evolved case, report **evolved law untested**, however successful the controls. If work is incomplete or invalid, report **incomplete/invalid**, not supported. A confirmed mismatch yields failure even if other cases remain to be completed. Arm B cannot rescue A.

The three scientific outcomes remain separate: **wrong route distribution → failed route prediction; no eligible evolved core → untested evolved law; eligible reproducible mismatch → failed law application**. There is no composite success score that hides any of them.

### 5. Scorer, random streams and prelaunch freeze

The 8086 machines, census classifiers, S2 runner and law/scoring orchestration are implemented and fixed-input tested. Actual immutable identities, independent review, validation, benchmark and cost gate are archived in the freeze annex below. L2 proof-coverage and specification limitations are explicit; this registration is not launch authorization.

| Frozen S2 path / responsibility | Existing structure to reuse and frozen behavior |
|---|---|
| `experiments/run8086.py` | `run6502.py::{exact_tapes,run_world}`, `algocell_exp/zsoup.py` (including `ZBridge`), `algocell_exp/soup.py`: cohort, cadence, exact counts, checkpoints and provenance with explicit 8086 machines. |
| `experiments/score8086.py` | `score6502.py::{read_input,load_snapshot,judge_tape,endpoints,score_world,report}`: input/hash validation, culture, chronological `t_rep`/`t_closed`, world tables and separate verdicts; add the full-horizon persistence pass. |
| Culture and trace execution | `algocell_exp/assay.py::assay_many`, `executor.py` with explicit machine selection; `census8086_trace.py::{DiagnosticMachine,mechanism,same_execution}` for ordered traces, mechanism and observation identity. |
| `experiments/law8086.py` | `census8086_law.py::{qualify,variants,assay}`, `census8086.py::panel`, and `LAW_CONTRACT.md`: separate extraction/freeze/intervention stages; actual evolved parents and source tracking, not reconstruction from a template. |
| `experiments/modal_s8086.py` and scoring wrapper | `modal_s6502.py`, `modal_score6502.py`, `modal_tier2_dispatch.py`: exact approved inputs, single-use containers, explicit resources, no automatic retries, append-only spend/provenance and refusal to overwrite results. |

Do **not** directly reuse `score6502.py::pair_has_loop` for the 8086: its non-6502 branch uses the old numeric-PC-decrease rule, and generic `pc_trace` assumes a `pc` field where 8086 exposes IP/CS. S1 taught both that sequential multi-byte wrap can look like a backward jump and that a forward alias transfer can return to an earlier ring position without a numeric PC decrease. Conversely, simple sequential ring laps are not automatically copy loops. The S2 mechanism verdict uses the existing full-address/start-and-provenance rule, not either shortcut. If extra control-flow diagnostics are reported, label them separately and account for actual decoded instruction length, full CS:IP, architectural seams, taken transfers, self-modification and retained REP. They cannot relabel a frozen `other` mechanism after outcomes.

Frozen random stream specification (implemented and byte-fixture tested before freeze): let

`K(arm, seed, L, purpose, tape_hex, case, generation)`

be SHA-256 of the UTF-8 encoding of the JSON array `["S2-8086-v1", arm, seed, L, purpose, tape_hex, case, generation]`, serialized with double-quoted JSON strings, `ensure_ascii=True`, separators `(',', ':')`, integer seed/L/generation and string case. Tape hex is lowercase, with no separators; unused tape/case is the empty string and unused generation is 0. Use the first eight digest bytes as a little-endian unsigned integer seed to explicit NumPy `Generator(PCG64(...))`; freeze NumPy/version and generated byte fixtures. Distinct purposes include `culture`, `route-trace`, `cells-final`, `law-rank`, `law-joint`, `law-serial`.

Culture uses that tape's `culture` seed in `assay_many(tape[None], n=64)` and its existing consecutive independent R/R2 draws. Route traces use `route-trace` to draw shape (16,L) uint8. Cache these results by arm/world/L/exact bytes, independent of sample or call order. Random-cell indices use `cells-final` with empty tape/case and `choice(20000,64,replace=False)`. Eligible random-cell genotypes are ordered by the full `law-rank` digest, ties by exact bytes. For each joint index 256…511, use `K(arm,seed,L,"law-joint",tape_hex,"joint:index",0)`, with the actual decimal index substituted, and draw exactly shape (d−6,) uint8 values uniformly from 0…255. Each serial case/generation draws a fresh shape (L,) uint8 partner from `law-serial`, with case IDs `single:position:value` and `joint:index`, substituting the actual decimal integers. This stream is independent of the census panel and of culture/tracing. Store actual partner bytes/hashes, not just RNG seeds. The constructed positive controls retain their original census streams as a separate panel.

Before launch, freeze and archive the approved S2 text in `REVISION_PREREG.md`, cohort/seed manifest, source hashes, dependencies, machine exports, RNG fixtures, scorer tests, partner panels, input schemas and projected cost ledger. Existing mechanism/law contract hashes are respectively `1aa291f608ffdd85493ce49718a65dde23bc6971f117b0267f4d7b9352c34193` and `028d41fa1fb60980cb3525ce12d548268d9783083adab142b32dfe708a985795`; the remaining source identities must be recorded from the final implementation rather than guessed here. Freeze before any S2 world runs, not merely before scoring it.

Required fixed-input checks: exact census replay for `B9 51`, `07 A5`, `FF 30`, `AD 50` and both phases; constructed MOVSB/MOVSW law controls; sequential ring laps; multi-byte offset wraps; forward alias returns; backward/taken branches; far transfers; retained REP; halts; self-modification; overflow→other; culture n=64 and trace n=16; sampling/tie/censor/persistence boundaries; complete partner/source/record identity. Inject known erasure, core and payload mismatches to verify rejection. Validate the shared endpoint structure against the archived Z80 DZ p=0 control (`results/s6502/SCORER_CONTROL.md`, corrected 10/10 parity), retaining its own historical partner streams and estimator. That regression does not claim S2's explicitly changed random-cell estimator is identical to the old n=32 one.

**No population pilot.** No short evolving soup, random-soup plumbing run, concentration trajectory or founder classification before registration. A future permitted throughput benchmark uses only predeclared fixed pair memories/active masks (frozen random-byte inputs, named census candidates and constructed REP cases), restored before every measured dispatch/repetition; outputs cannot feed a subsequent population step. Measure execution equivalence and timing of the actual dispatch, restore, sample/readback and serialization components, including the fixed law workload. Archive the benchmark inputs and timing; no population-selection information is available. It may choose hardware, resource bounds, batching/submission settings and timeout, and refine the spend forecast. It may not tune biological arms, seeds, mutation policy, horizon, cadence, predictions, thresholds or eligibility from a population observation. A material design change requires a dated replacement proposal and approval before population work.

### 6. Spend estimate and hard stop

All rates here come from saved records; this is not a new price quote or an 8086 Modal measurement. No credits are assumed. “Step” in this section means one **world-step**, not one instruction or REP iteration in one encounter.

#### Actual records and relevant measurements

| Evidence | What it establishes |
|---|---|
| S1 `results/s6502/SMOKE.md`, `smoke_check.json`, `LAUNCH.md`; `modal_s6502.py` | Measured L40S execution: 4,842.705 world-steps/s for A=L16/128, 4,623.173 for B=L32/128, 2,678.259 for C=L16/512. S1's $140.58 for 80×10M is a launch **projection**, not a final population invoice. At A/B performance, 40×3M with the truncated cadence would cost about $18.43 on L40S including the recorded overhead allowances and 25% margin; this is a 6502 reference, not an 8086 forecast. |
| S1 correction `results/s6502/v2/cost.csv`, `cost_summary.json`, `modal_calls.json`, `modal_apps.json` | Rescoring 80 S1 worlds plus ten controls: $0.09818 function compute estimate, $0.87049 whole-app resource-time bound, $0.88049 with storage/egress. These are bounds/estimates for rescoring, not population costs or all-value evolved-law costs. |
| T2 `results/tier2/SPEND.md`, `run_spend.json`, `score_spend.json` | Actual app-attributed metered worlds/panels $19.16065586 plus scoring $0.39581494 = **$19.55647080**. Billing can lag; function-body estimates are not the bill. |
| T2 `DESIGN_FULL.json` plus StageE-L10 in `tier2_dispatch.py` | 253.4M soup world-steps. Crude worlds/panels cost $0.0756143 per million world-steps, or $0.0771763/M including scoring. These divide mixed conditions and fixed panel/scoring overhead by soup steps; use only as an order-of-magnitude cross-check. |
| T2 `results/tier2/l4.json`, `COST.md`; `modal_tier2_dispatch.py` | Slowest fresh-state L4 repetitions: L16 3,223.6862 and L32 3,306.9067 world-steps/s. Costing reduces throughput a further 20%, then adds 10% runner overhead. Proposed allocation is one CPU and 8 GiB RAM per L4 container. |
| Saved `results/s6502/v2/modal_rates.json` | L4 $0.80/GPU-hour + CPU $0.04730/core-hour + RAM $0.008/GiB-hour; proposed allocation totals **$0.91130/container-hour**. |

The supplied M4 core comparison, 334M versus 1,091M execution steps/s, is a ratio of 3.2665 (rounded to 3.3 for the sensitivity budget). It cannot simply multiply a Modal population bill: the 20,000-cell/8,192-pair pipeline has five small dependent dispatches and is latency-bound (`planning/ZILION_DESIGN_2026-10-10.md` §2.4). If only fraction f of wall time were slowed by 3.3, the elementary model gives `1+2.3f`, not necessarily 3.3. This is an explanatory model, not a fitted bound; GPU register pressure, divergence and encountered instructions can change f and throughput.

The saved Modal measurements illustrate the distinction: T2's fresh L16 minima on L4 and L40S are 3,223.69 and 3,607.94 world-steps/s, only about 12% apart despite their price difference. S1's fourfold instruction-budget increase slowed execution about 1.808×, not 4×. Neither measures 8086 performance. **There is no saved 8086 Modal benchmark in the inspected records.** Gate 2's local census timings do not fill that gap. Translating the M4 ratio on Modal therefore remains a fixed-input prelaunch requirement, not a claimed completed check.

#### Proposed dollar envelope

| Calculation | USD |
|---|---:|
| `(60M/3223.6862 + 60M/3306.9067) / 3600 × 0.91130`, matched Z80 L4 timing baseline | 9.3044 |
| Baseline `/0.8 ×1.10`, T2 throughput and runner margins | 12.7935 |
| Apply 3.3 to that **whole pipeline** as a conservative sensitivity scenario for 8086 population work | 42.2187 |
| Complete culture/route/extraction/all-value/serial/control scoring reserve | 10.00 |
| Fixed-input validation and Modal timing reserve | 2.00 |
| Startup, compilation, initial snapshot, storage and egress reserve | 1.00 |
| **Provisional total** | **55.22** |
| **Hard cap for preparation + all 40 worlds + all required assays** | **60.00** |

The population scenario is about 46.33 aggregate container-hours, or 69.5 minutes/world on average. It is not a timeout recommendation or a proven worst case. The $10 assay reserve is about 10.97 L4-container hours and is a proposed allowance, not a measured guarantee. At the ≤640 evolved genotype-world case ceiling, the maximum law workload is 81,920,000 encounters: up to 76,800 per L16 case and 179,200 per L32 case, with 320 cases of each. The ten constructed controls add 1,185,792, for **83,105,792 law encounters** before diagnostic replays; culture, route classification, extraction and output processing are additional. The complete fixed-input cost gate must account for that workload, not just cheap endpoint rescoring. Deduplicating identical computations may save cost but cannot drop required challenges or denominators. The crude T2 metered rate would put 120M comparable steps near $9.07 before the 8086 allowance, consistent with the unadjusted baseline.

Use the fixed-input benchmark to project the **complete** frozen design, including longest-lived containers, startup and remaining assays, on the chosen resources. L4 is the proposal because of the latency-bound comparison; L40S may be selected only if its measured all-in forecast fits the same cap. Verify then-current rates in the approved preparation step. If the complete forecast exceeds $60, do not launch populations; return a revised spend proposal. Do not silently reduce worlds, horizon, sampling or assays to fit.

The eventual dispatcher must reserve running liabilities as well as remaining work. Track app-attributed metered spend and a conservative resource-time accrual including startup/idle time; billing lag cannot authorize overruns. Before dispatching a job, reserve its remaining allocation and all required unstarted work. Checkpoint/stop active work early enough that the resource-time ceiling, termination lag and storage reserve cannot exceed $60. No automatic retries. A cap-triggered stop is an **incomplete experiment**, not administrative biological censoring or a scored negative. The user must approve any higher cap; this draft does not spend the proposed $60.


### L2 — the broader evolved-law family (secondary, separate verdict; never rescues or replaces L1)

**Eligible:** any culture-positive evolved candidate whose frozen code-and-ordered-trace certificate shows a forward constant-displacement copy kernel: same-valued byte reads at source positions s + k paired with writes at s + d + k, in increasing k on the pair ring, with constant d, c ≤ d ≤ L and d | L; a d-periodic baseline parent at entry phase zero; no original-partner reads, interfering writes or payload-dependent address, count or stop conditions; full partner coverage; and an offspring that resets into the same admissible family. The kernel may be REP MOVS of any setup or any instruction loop meeting this definition.

**Core footprint C:** positions in the primary seed 0 … d − 1 that the certificate trace shows are fetched as instruction stream (opcode, prefix, operand or displacement), or read as data feeding an address, count or condition. Determine C from code and trace before any retention assay, never from observed sites. Byte-register dependencies retain the unchanged high/low byte. Original erased-coordinate code/control dependencies cannot be silently reduced modulo d into held-fixed primary core.

**Prediction:** exactly positions in {0 … d − 1} \ C are free payload coordinates; every position d … L − 1 is erased; the core and displacement are inherited exactly. L1 cases also appear in L2. Report L2 both with and without L1 cases, with independent denominators and verdicts. No L2 result rescues or replaces L1.

**Assays and rules:** use the same candidate register, culture, independent certificate/data freeze, interventions, zero tolerance and three-outcome discipline as L1. Apply the ≤16 distinct-genotype cap separately per family per world, preserving all eligible first/closure/final-top priorities and filling from the keyed random-cell order. For a noncontiguous C, order free coordinates increasingly; joint payload index j enumerates that order. The counterbalanced assignment is (37*j+v) mod256, and random assignment shape is (|free|,) under the same frozen keys. Hold every core coordinate fixed, change only the primary seed coordinates, retain erased controls and all17 census partners, and serially pass actual B output through8 fresh-partner generations. L1/L2 overlap may be cached only for identical input/challenge/geometry and retains both memberships. The forecast assumes no duplicate-execution savings.

**Validation and proof scope:** all ten constructed MOVSB/MOVSW controls reproduce C={0,…,5}. The constructed word LODS/STOS/LOOP control at L32/d16 reproduces C={0,…,9}, canonical primary source ancestry and reset closure. The extractor code is frozen before any world. It uses explicit byte-dependency transfers and semantic read/write pairing; unsupported instruction forms remain unresolved, are published and suspend a supported L2 arm verdict. Complete automatic proof extraction for arbitrary 8086 loops is not implemented; this is not silently treated as exclusion from the entire approved broad family.

**Known specification defect, found before launch:** the approved eligibility permits nonzero source start s, but its prediction names primary0…d−1. The reviewed and fixed-input GPU-verified L32/d16 parent whose seed begins `BE 0C 00 BF 1C 00 49 F3 A4` starts at s=12 and copies with displacement16. It qualifies at baseline, has C={0,…,8}, full B coverage, no original-partner reads and reset closure. Its actual original sources are12…27: predicted free primary9…11 are erased and excluded original25…27 survive. The canonical source prediction is therefore already false for an allowed constructed input. This is a known design defect, not an evolved result. The registration and code preserve the approved definition literally; they do not impose s=0, shift/repair the seed or exclude the input based on a retention outcome. A source-origin restriction or shifted-seed law requires a prospective design resolution by the coordinator/user before any population launch. See `results/s8086/prelaunch/SPEC_LIMITS.md` and the independent review.

