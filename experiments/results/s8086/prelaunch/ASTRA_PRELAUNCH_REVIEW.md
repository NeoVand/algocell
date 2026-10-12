# Astra independent prelaunch review — S2 / 8086

2026-10-12. Reviewed branch `feat/s2-8086`, HEAD `df3d4b8`, before any population launch. This is a review and prospective amendment proposal, not a release, registration change, or spending authorization.

**Decision: do not launch this freeze.** The forecast exceeds the approved cap, the registered L2 prediction has a known fixed-input counterexample, and this review finds two additional mechanism-classifier defects and a gap in the physical spend bound. The chronological endpoint and sampling logic largely implements the registration correctly, but the mechanism labels cannot yet support their advertised interpretation. The existing launch gate correctly keeps this freeze stopped.

Only this report was written. No Modal calls, GPU work, population observations, source edits, or commits were performed. Read-only parallel audits covered the scorer, runner/guard, and historical cost/timing evidence; their findings were reconciled here. All 131 `freeze.json` file hashes match the worktree. Eight focused existing CPU tests passed with bytecode and pytest caching disabled. Additional CPU checks used archived certificates and small synthetic traces; those checks do not constitute new machine/GPU validation.

## 1. Adversarial findings

Locations below are relative to `experiments/`, at the reviewed HEAD. BLOCKER means resolve before releasing the affected confirmatory experiment; SHOULD-FIX means a concrete robustness or reporting defect; NOTE identifies a material scope limit or a check that passed. The earlier `INDEPENDENT_REVIEW.md` sign-off does not close the new findings below.

| ID | Severity | File and line | Finding, consequence, and required disposition |
|---|---|---|---|
| B1 | **BLOCKER** | `results/s8086/prelaunch/FREEZE_REPORT.md:20–36`; `s8086.py:82–90` | **The frozen design is not fundable at the approved cap.** $138.721091 exceeds $60 by $78.721091. This is an all-in forecast, not a GPU-only subtotal. The gate correctly refuses it. Select and approve a prospective numerical amendment or a higher cap, then regenerate the complete forecast and freeze; a mid-run cap stop cannot become a biological negative. |
| B2 | **BLOCKER** | `REVISION_PREREG.md:1633–1643`; `law8086.py:138–146,280–299,374–375`; `results/s8086/prelaunch/SPEC_LIMITS.md:5` | **L2 eligibility permits a source origin that its prediction does not describe.** The s=12 control is eligible, but `variants()` mutates primary 0…d−1 and treats d…L−1 as erased; `check_descendant()` demands ancestry `q % d`. Actual roots are 12…27. A reproducible rejection here would reproduce a known specification defect, not discover an evolved-law failure. Adopt an explicit source-root/phase amendment, including the core-versus-erasure distinction in §2, and update extraction, interventions, diagnosis, serial ancestry, and reports together. L1's s=0 theorem is unaffected. |
| B3 | **BLOCKER** | `algocell_exp/census8086_trace.py:38–44,145–147,231–235`; `algocell_exp/machines/i8086_pair_maps_L16.wgsl:313,319–320` | **The observer does not consistently record full instruction-start addresses.** It injects `i8086_pc()` into the start hook, ignoring the hook's `pc` argument. Ordinary execution has already fetched the opcode and advanced IP; retained REP has not. Mixed address conventions can miss or invent a repeated start. This violates the registered loop criterion even though full original records and memory remain identical. Use the actual start argument and revalidate the boundary fixtures and affected census labels before refreezing. See the concrete CPU reproduction below. |
| B4 | **BLOCKER** | `algocell_exp/census8086_trace.py:166–186,198–201,221–228`; `algocell_exp/machines/i8086_pair_maps_L16.wgsl:384`; `results/census8086/MECHANISM_CONTRACT.md:11–18`; `score8086.py:62–77` | **A changed memory-derived register value can be labeled an exact copy.** Implicit register arithmetic leaves stale provenance, and a stored byte need only occur somewhere in the saved read-value list. `MOV SP,[14]; PUSH SP` supplies a concrete false positive: the arithmetic-modified low byte is matched to the previously read high byte. Audit all implicit SP/SI/DI/CX updates and track actual byte pairing, not value membership. This affects mechanism distributions and the meaning of R2's memory-fed numerator. Reaudit affected census classifications; replaying an old wrong label is not sufficient validation. |
| B5 | **BLOCKER** | `dispatch8086.py:102–111,139–150`; `modal_s8086.py:28,38–39,62`; `modal_score8086.py:20–21,64,66–67,88`; `results/s8086/prelaunch/benchmark_failed_attempt.json:10–19` | **The reservation is not a hard bound on complete container/app lifetime.** The dispatcher lowers input execution timeouts, reserves `(deadline+120)*RATE`, and then waits in unbounded `call.get()`. Startup/import attempts and infrastructure restarts are not bounded by that 120-second allowance. The archive already records repeated imports despite `retries=0`. Add explicit startup limits and a lifecycle watchdog independent of the blocking dispatcher process that cancels/stops the app before the reserved resource-time envelope expires. Account for attempts and termination; retain unknown liabilities until stopping is confirmed. Reforecast afterward. No population overrun is alleged; the current cost gate still blocks launch. |
| S1 | **SHOULD-FIX** | `dispatch8086.py:54,69–71,83`; `meter8086.py:31–34` | **New metering does not update the reported liability or enforce the cap in that transaction.** `total` is calculated before the incoming `metered` event; only cumulative monotonicity is checked. The standalone meter writes a report but dispatch never polls or incorporates it. A CPU example with a $59 plan accepts a new $60 meter and still returns $59; only the next reservation notices. Apply the new meter before recomputing the ceiling and connect it to dispatch/watchdog actions. Billing lag still requires an independent resource-time bound. This is part of repairing B5, not a replacement for it. |
| S2 | **SHOULD-FIX** | `run8086.py:81–93,119–124`; `modal_score8086.py:35–43,61`; `score8086.py:107–112` | **Some technical stops lack an explicit terminal failure record.** Initialization and the first commit precede the world handler; that handler then depends on another GPU read, compression, file creation, and storage commit. A broken device or storage operation can mask the original error and prevent the `incomplete` row. Scoring may leave only a reservation marker and remote error. Persist a host-side terminal record with original failure, call identity, and last acknowledged progress independently of the failing GPU/storage path. The present scorer rejects missing summaries/samples, so this is a diagnosis/provenance gap rather than silent biological censoring. |
| N1 | **NOTE** | `law8086.py:18,90–91,114–115,171–173,245–269`; `REVISION_PREREG.md:1641` | **L2 is not an implemented proof procedure for every eligible 8086 loop.** Unsupported forms remain unresolved and suspend support for an otherwise complete arm, including when other cases pass. This is appropriately disclosed. Report unresolved extraction counts and independent evolved-world coverage; never convert them into biological negatives, silently narrow the family, or call constructed success an evolved pass. |
| N2 | **NOTE** | `law8086.py:296–324,326–331,349,374–375` | **Successful assay rows use the certified static source map, not a new per-mutant ancestry trace.** This is valid only if the independent dependency proof establishes invariant execution and source pairing under the permitted interventions. Value equality alone is insufficient. Mismatch replay is helpful but cannot expose a falsely certified ancestry change that preserves bytes. The L2 amendment must preserve the proof obligation and add source-map/phase tests; it cannot merely change the expected byte array. |

### Concrete new classifier counterexamples

**Start-address error (B3).** The archived GPU-produced `certificate_REP_L16_d8_a4.json.gz` begins with type-4 start events `(step,address) = (0,1),(1,4),(2,5),(3,4),…`; the actual starts are 0,3,4,4,… for `MOV DI,8; DEC CX; REP MOVSB`. The source explains the discrepancy: ordinary `i8086_execute()` saves `pc`, fetches, then calls `on_instruction_start(pc,op)`, while the injected observer rereads the advanced PC.

A kernel-consistent CPU synthetic trace for L16 `BF 10 00 B9 02 00 F3 A5 F4` (`MOV DI,16; MOV CX,2; REP MOVSW; HLT`) supplies a decision-changing boundary case. Frozen logged starts `[1,4,7,6]` yield **memory-fed one-pass copier**; true starts `[0,3,6,6]` yield **confined copy loop**. This is a mechanism counterexample, not evidence that this partial writer passes the culture gate. Observation parity over the original record cannot validate the meaning of newly appended diagnostic fields.

**Changed-value provenance error (B4).** Put `8B 26 0E 00 54 F4` at L16 positions 0…5 and `14 12` at positions 14…15. `MOV SP,[14]` loads `0x1214`; this kernel's 8086 `PUSH SP` stores the decremented value `0x1212` at pair-ring positions 18 and 19. The classifier retains read-values `[0x14,0x12]` and accepts both stored `0x12` bytes by membership, returning **memory-fed one-pass copier**. The low byte is actually arithmetic-derived. This reproduces the contract violation without invoking the GPU. Neither example demonstrates a population threshold flip; both refute the promised semantics for arbitrary evolved inputs.

### What did survive review

- **Chronology and the S1 early-exit failure:** `score8086.py:14–42` scans every scheduled sample even after both first events. It applies the 0.005 share gate, largest-share/lexicographic tie rule, 11-point mechanism persistence, and separate same-genotype indicator. `run8086.py:94–118` has no success-dependent stop. Missing data cannot count as a completed 3M negative.
- **Endpoint meanings:** R0/R1 and `t_closed` use culture and full-encounter fetch, independently of loop classification. Closure does not require a loop or law eligibility. B3/B4 primarily compromise the semantic mechanism tables and R2; they do not by themselves refute the raw fetch-based endpoints. Open/intermediate/confined remain separate, and R2 requires the same ≥9 encounters to be open and memory-fed (`score8086.py:58–77`).
- **Partner-source mismatch:** culture uses 64 fresh uniform partners in each of two generations; every first-generation offspring proceeds. Route traces use their separate 16 uniform partners; laws use zero plus the frozen census panel and separate serial partners. They are not replaced by soup partners. The archived DZ regression deliberately retains its historical estimator/streams rather than claiming identity with S2's changed random-cell estimator (`score8086.py:44–84`; `algocell_exp/assay.py:296–336`; `law8086.py:335–370`).
- **RNG and multiplicity:** the JSON/SHA-256/first-eight-little-endian/explicit-PCG64 construction matches the registration, with arm/purpose/tape/case/generation separation (`s8086.py:31–45`). Final cell indices precede genotype/culture inspection; multiplicities remain in the 64-cell denominator and deduplication is for execution (`score8086.py:153–171`; `law8086.py:189–211`). Same-number arm seeds are not treated as matched worlds. Inherited soup initialization and parameter-ring stepping preserve the registered streams; hardware collision/mutation races still preclude bitwise population replay promises.
- **Missing information and verdict separation:** unassessable qualifying top tapes make the world unscoreable. They cannot be skipped to invent a later first event or censored absence. Both route and law support require all 20 registered, scoreable worlds in the arm; A and B remain separate. Empty eligibility means **evolved law untested**, not supported (`score8086.py:23–27,190–210`; `law8086.py:235–272,411–415`).
- **Normal ledger bookkeeping:** the hash chain, lock, full unstarted-work reservations, and retained failed-call liabilities are real safeguards (`dispatch8086.py:24–83`). B5 concerns the physical duration represented by those reservations. Modal 1.6.1 is the frozen dependency. Its installed static startup default inherits the decorator timeout; the per-call `with_options(timeout=...)` path does not explicitly reduce that separate startup allowance. Current official [timeout documentation](https://modal.com/docs/guide/timeouts) separates startup and execution, and [function documentation](https://modal.com/docs/guide/functions) distinguishes infrastructure retry behavior from configured exception retries. The preparation retry record makes this an observed platform behavior, not merely a hypothetical concern.

## 2. Prospective L2 amendment

The following is proposed replacement text, to be approved and frozen with corresponding code before any population work. It changes neither L1 nor the route endpoints. It preserves the existing within-one-period core-footprint bound, avoiding an unrelated expansion of L2 proof coverage.

> **L2 source-root, core, and phase amendment.** Let A coordinates be `0,…,L−1` and pair-ring coordinates be `0,…,2L−1`. The actual sampled baseline parent must be d-periodic at physical entry phase zero. The certified forward kernel copies bytes in the ordered sequence `s+k → s+d+k (mod 2L)`, with `d | L`, the existing `c ≤ d ≤ L` footprint bound, full B coverage, and no original-partner reads or interfering writes. Establish the actual original-A source roots from the code and ordered trace before payload assays. For this family the root interval is `R={s,…,s+d−1}⊆A`; certify this interval and the complete source map rather than infer it from equal byte values or a cyclic similarity offset. Inputs whose roots/geometry cannot be established remain unresolved or outside this stated family, with the reason recorded.
>
> Let `K⊆{0,…,d−1}` be the independently certified raw core footprint: original coordinates whose values supply executed instruction bytes or data governing addresses, counts, conditions, or stopping. Keep the existing footprint bound `c=1+max(K)≤d`; K can be noncontiguous and is never inferred from measured retention. Record raw original dependencies without reducing erased-coordinate dependencies modulo d. Unsupported dependency transfers remain unresolved.
>
> For physical offspring coordinate q in B, define
>
> `r(q) = s + ((L + q − s) mod d),       0 ≤ q < L.`
>
> Require the independently extracted ordered source map to equal r across the baseline challenge panel. Define the **source core** `C_R={r(k):k∈K}`, the **free source coordinates** `F=R\C_R`, and the **ancestrally erased coordinates** `E=A\R`. Hold both the entry core K and its inherited source core C_R fixed. The certificate must establish that arbitrary assignments at F, and the specified erasure substitutions at `E\K`, cannot change the kernel, source map, entry-phase core, stopping behavior, or reset closure. Agreement of 17 baseline outputs alone is not this noninterference proof.
>
> **Prediction:** for a permitted input mutant x, the complete first offspring is exactly `B[q]=x[r(q)]`, with original-source identity r(q), at the physical B entry phase zero. Exactly F are independently variable source payload coordinates; their count is `d−|C_R|=d−|K|`, not necessarily `d−c`. Every original coordinate in E is absent from offspring ancestry. Core and displacement values at the declared physical entry positions are preserved exactly. If a setup coordinate lies in `K∩E`, its original ancestry is erased even though its value is needed to run the parent; its descendant core value comes from C_R.
>
> **Phase and alignment:** write the actual source seed as `S[j]=x[s+j]`, `0≤j<d`. Then `B[q]=S[(q+L−s) mod d]`. The offspring has phase `(L−s) mod d` in this source-seed indexing, while execution/reset remains at physical B coordinate zero. Compare at that fixed physical alignment: no search for a favorable rotation, no parent rotation/repair, and no replacement by a tiled prefix. Since `d | L`, `r(q)≡q (mod d)` and r is the identity on R. Thus `r(r(q))=r(q)`; repeated transfer retains the same generation-zero source identities under the certified reset kernel.
>
> **Interventions:** enumerate F by increasing physical coordinate. Test all 256 values at each free coordinate, plus the same 256 counterbalanced and 256 keyed random joint assignments, using that order for j. Counterbalanced values are `(37*j+v) mod 256`, for `v=0,…,255`; retain the registered keyed random streams. Change only F; hold K and C_R fixed and leave other coordinates at their sampled baseline values. Challenge against all 17 frozen census partners and serially transfer the actual B offspring through eight fresh-partner generations, checking complete output and generation-zero ancestry under r at every transfer. Test every coordinate of `E\K` through all 256 values as single erasure controls: expect the baseline offspring and absence of that original source. Coordinates `E∩K` are held setup/core controls, not arbitrary-value erasure controls; publish their count and certify loss of their original ancestry in the unchanged-core execution. An arbitrary alteration of an executed setup byte is not a payload-neutral intervention.
>
> Candidate sampling, family-specific ≤16 caps, pre-intervention eligibility/data freeze, zero tolerance, fixed-input diagnosis, unresolved-proof handling, and separate L2-with/without-L1 verdicts otherwise remain unchanged. No retention outcome changes eligibility or selects a new alignment. L1 remains its original six-byte, s=0 law.

### Checks against the frozen controls

For **L32/d16/s12**, the archived certificate has `K={0,…,8}`. The amended quantities are:

| Quantity | Prediction |
|---|---|
| Actual roots R | `12…27` |
| Source core C_R | `16…24` |
| Free source coordinates F | `{12,13,14,15,25,26,27}` — seven coordinates |
| Ancestrally erased E | `{0,…,11,28,…,31}` |
| All-value erasure controls E\K | `{9,10,11,28,29,30,31}` |
| Held but ancestrally erased setup E∩K | `0…8` |
| Physical B source map | `[16,…,27,12,…,15]` repeated twice |
| Phase in the source seed starting at 12 | `4 mod 16`; physical offspring entry remains zero |

Thus primary 9…11 are erased and originals 25…27 survive, as SPEC_LIMITS reports. Original coordinates 16…24 supply the offspring's core at positions 0…8. The output's complete source map is idempotent under serial composition. In particular, blindly changing the prediction to `R\K` would falsely call all 16 roots free: it would allow mutation of the inherited core at 16…24. Blindly testing every E byte as a neutral erasure would also be false, since E includes the executing setup.

For every constructed **s=0 REP** control, `K=C_R={0,…,5}`, `r(q)=q mod d`, `F={6,…,d−1}`, `E={d,…,L−1}`, and phase is zero. The amendment therefore reduces exactly to L1's existing predictions and interventions:

| L | d | Free coordinates per MOVSB/MOVSW control | Erased coordinates |
|---:|---:|---:|---:|
| 16 | 8 | 2 | 8 |
| 16 | 16 | 10 | 0 |
| 32 | 8 | 2 | 24 |
| 32 | 16 | 10 | 16 |
| 32 | 32 | 26 | 0 |

The constructed L32/d16 non-REP LODS/STOS/LOOP control similarly retains K=0…9, six free coordinates, canonical roots, and zero phase. CPU inspection checked the amended r against every archived partner source map for the s=12 control and all ten REP controls, including `r∘r=r`; the non-REP archived map also agrees. These are source-map/algebra checks against already archived GPU evidence. The new nonzero-source intervention and serial suite has **not** been executed and must be validated before refreeze.

**Does eligibility need changing? Yes, explicitly in its core/noninterference interpretation.** Do not add `s=0` or `K⊆R`: either would discard the admitted counterexample rather than describe it. Retain forward geometry, periodic baseline, original-partner exclusion, full coverage, footprint bound, and reset closure. Replace any blanket requirement that an ancestrally erased coordinate cannot influence setup with the precise distinction above: unprotected payload/erasure coordinates cannot influence execution; held `E∩K` setup coordinates may. Require C_R to preserve the core at the physical offspring phase and require the same proof after transfer. “Core inherited exactly” means the exact entry-phase core values, not necessarily ancestry from the original setup copies in K. Keep raw dependency identities separate from generation-zero ancestry during diagnosis. This is a prospective eligibility/assay clarification, not permission to admit new K footprints or silently reduce raw dependencies modulo d.

Implementing only a shifted `free` list is insufficient. `law8086.py:138–146,280–299,310–312,374–388` and the certificate/denominator schema must consistently represent K, C_R, R, F, E, held-erased coordinates, fixed phase, and composed ancestry. Keep the conservative existing encounter ceilings until a revised implementation is measured; do not assume savings from fewer erasure controls.

## 3. Cost options and scientific consequences

### Forecast method

These are mechanical reprojections of the freeze's measured components, not new benchmarks or hard guarantees. The frozen L4 rate is `$0.00025286/s` including one physical CPU and 8 GiB; minima are 350.202034 world-steps/s for A/L16 and 425.418704 for B/L32. Preserve the frozen 20% throughput reduction, 10% runner margin, cold/startup allowances, scoring/extraction factors, and complete assay ceilings (`benchmark8086.py:175–224`; `results/s8086/prelaunch/projection.json`). B5 still requires a repaired enforceable lifecycle bound before any option can launch.

For horizon H, truncate the registered sampling schedule at H and retain distinct checkpoints through H plus the final checkpoint. There are 585 samples/5 checkpoints at 3M, 505/4 at 1M, and 525/5 at 1.5M. Recompute each arm's world allocation and the `3×samples+69` score ceiling. Extraction and both separate law panels remain fully reserved per retained arm; shorter time does not reduce the 64-cell sample or the ≤16-case-per-family cap.

In notation matching the frozen projector, with rate r, measured minimum v, sample time a, snapshot time b, culture/route time u, and `startup=max(120,2×compile_s)`:

`world_arm = 20 r [ceil(H/v/0.8×1.1 + n a×1.5 + k b×1.5 + startup + 60) + 120]`

`score_arm = 20 r [(3n+69) u×1.5 + startup + 180]`.

Every option retains the **entire $6.119213 fixed allowance**: controls $0.117248, diagnosis $0.202061, already incurred preparation bound $0.799904, and storage/export/termination reserve $5. This conservative comparison does not halve the global reserve, drop any of the ten controls, or assume L1/L2 caching savings. Totals include preparation; they are not incremental new-spend figures.

| Option | Population worlds | World | Score | Extract | Laws L1+L2 | Fixed | Complete forecast | Against $60 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **(a) Frozen** | A+B, 20 each × 3M | $111.880436 | $5.749245 | $3.211350 | $11.760849 | $6.119213 | **$138.721091** | $78.72 over |
| **(b)** | A, 20 × 3M | $61.202234 | $2.816378 | $1.605106 | $4.284714 | $6.119213 | **$76.027646** | $16.03 over |
| **(c), 1M** | A+B, 20 each × 1M | $39.441103 | $5.392018 | $3.211350 | $11.760849 | $6.119213 | **$65.924532** | $5.92 over |
| **(c), 1.5M** | A+B, 20 each × 1.5M | $57.550936 | $5.481325 | $3.211350 | $11.760849 | $6.119213 | **$84.123672** | $24.12 over |
| **(d)** | A, 20 × 1.5M | $31.405212 | $2.688166 | $1.605106 | $4.284714 | $6.119213 | **$46.102411** | $13.90 nominal headroom |

Displayed components are rounded independently; totals use full precision. A-only is not half the complete forecast: L16 population work is slower, whereas L32's maximum law workload is larger. Both arms at 1M still exceed the cap because assays, extraction, startup, and output liabilities do not shrink in proportion to world-steps.

### Historical evidence for horizon adequacy

The useful comparison is **arrival versus closure versus eligible-law appearance**, not throughput versus biological time. GPU speed changes dollars per world-step, not the meaning of a horizon in world-steps.

| Historical cohort | Arrival evidence | Closure evidence / horizon sensitivity |
|---|---|---|
| Z80 G, L16 | Ordinary median first detection 525 steps | KM first closure 21,000; all 20 closed by 143,000, within its 300k horizon. |
| Z80 K, L32 | Ordinary median first detection 425 steps | KM first closure 813,000; only 12/20 closed by 1M. Early arrival did not imply rapid closure. |
| Corrected 6502 A, L16/128 | KM first detection 1.125M | KM closure also 1.125M. Arrivals/closures: 9/20 by 1M, 12/20 by 1.5M, 19/20 by 3M, 20/20 by 3.55M. |
| Corrected 6502 B, L32/128 | KM median not reached by 10M | Only one arrival/closure by 3M; four by 10M. |

Sources: `results/review_r2/FIRST_CLOSURE.md:3–16` and its underlying world table; `results/s6502/v2/S1_v2.md:8–13`; recount of `results/s6502/v2/s1_worlds.csv:2–41`. The Z80 arrival column uses ordinary medians, as labeled in the historical table; it is not being substituted for a KM estimate. These protocols have different machine, historical scorer, and random-cell-law identities. The comparison is contextual and a horizon sensitivity argument, **not an 8086 arrival-rate prediction** (`REVISION_PREREG.md:1446,1508`).

The corrected 6502 report has **0/0 final offset-law-eligible cases despite many closures** (`results/s6502/v2/S1_v2.md:10–13`). Closure is not sufficient for a specific law family. S2's narrow L1 REP template, periodic baseline, finite candidate register, and conservative L2 proof coverage impose additional filters. Consequently even 3M cannot guarantee an evolved law test; reducing the horizon increases the risk of **evolved law untested**, and an unresolved certificate can instead force **incomplete/invalid**. Constructed controls cannot repair either gap.

| Option | Scientific cost to route verdicts | Scientific cost to law verdicts |
|---|---|---|
| (a), both 3M | Preserves all registered A and separate B questions, including the 2.75–3M persistence interval. Budget makes it unavailable as frozen. | Retains the most time and both length scopes among these choices; still no guarantee of eligibility or resolved proofs. |
| (b), A 3M | Preserves the complete primary A experiment and all its original endpoints; withdraws the B scope test. | Preserves A's opportunity for later closed/periodic cores and final sampling; loses B's longer-tape cases and any L32 scope claim. Scientifically preferable to shortening A, but over cap. |
| (c), both 1M | Preserves 20 worlds per length but changes the finite-horizon question. Highest risk of insufficient D, missed late memory-fed events, and observing an early persistence interval only. The 6502 A sensitivity would have D=9, below S2's R0 target. | Strongest truncation of post-arrival maturation and final-core sampling. Z80 L32 and 6502 timing do not justify calling 1M generally adequate. Two thin panels do not rescue an empty A panel. Still over cap. |
| (c), both 1.5M | Less truncation than 1M, but loses all 1.5–3M route opportunities and the registered late interval; B remains a separate test. | More opportunity than 1M, materially less than 3M. The 6502 A contextual count is 12 rather than 19 by 3M, before any law-eligibility filter. Still over cap. |
| (d), A 1.5M | Preserves all 20 primary seeds, but sacrifices B and late A events. Results address only a prospectively amended 1.5M question. | Accepts a greater risk of no testable evolved core or narrow world coverage. Budget fits with nominal room for repair/revalidation; law support must retain its conditional, panel-limited wording. |

**A shorter run cannot be scored under the existing 3M registration.** A prospective amendment must move the occurrence limit, censoring horizon, final-top and 64-cell snapshot, required completeness checks, schedules/schemas, and forecast together. R2b needs a newly stated interval, for example exactly 11 points `H−250k,H−225k,…,H` at 25k spacing. For H=1.5M that is 1.25–1.5M; for H=1M it is the 11-point 750k–1M subset of the denser archived schedule, not all 101 samples in that interval. These are changed estimands, not technical stops or evidence for the original 2.75–3M claim. Retain the same 20-world integer cutoffs unless a separately justified prospective amendment changes them; do not tune them after observing S2.

### Hardware and a ≤$1 decision benchmark

At the checked [Modal prices](https://modal.com/pricing), one L40S plus the same CPU/RAM costs `$0.00057286/s`; one H100 costs `$0.00112786/s`, versus L4's `$0.00025286/s`. They need approximately **2.2655×** and **4.4604×** L4 throughput merely to lower time-proportional cost. For the measured A rate, that means about **793.4** and **1,562.1 world-steps/s**; for B, **963.8** and **1,897.5**. Startup and law/scoring performance must also enter the full forecast.

A saving is plausible, but a larger GPU is not a demonstrated remedy. The saved Z80 minima were 3,223.69 steps/s on L4 and 3,607.94 on L40S, only 1.12× (`planning/PREREG_8086_DRAFT.md:222–224`). The five dependent dispatches and synchronization limit gains from more GPU capacity (`planning/ZILION_DESIGN_2026-10-10.md:584–610`). Conversely, 8086's roughly ninefold slower L4 rate leaves room for a larger execution, register-pressure, or divergence component than Z80 had. Without profiling/timing, neither pure latency dominance nor a cost-saving speedup is established. Peak FLOPs and bandwidth alone do not decide this workload.

A **prospectively authorized, ≤$1 fixed-input probe** would decide whether either GPU crosses cost break-even and whether a complete repaired forecast fits the chosen cap. For example, reserve $0.25 for L40S, $0.50 for H100, and $0.25 for shared setup/termination/storage. Convert those reservations into external lifecycle deadlines at each resource rate, covering startup/restarts rather than only the function body; stop without retry when the envelope expires. An inadequate or failed probe is inconclusive, not permission to exceed $1.

Use the frozen restored pair memories, masks, named witnesses, REP geometries, non-REP control, and mixture; outputs must never feed another population step. Validate full raw memory/lane equality, corrected diagnostic semantics, storage-observer behavior, and RNG fixtures. Measure cold repetitions of the actual ≤50-step submission/readback path, sample/checkpoint costs, both law workloads, and logging/gzip overhead with the same conservative margins. Compare **all-in projected dollars**, not just kernel throughput. No such benchmark was run for this review. It would select hardware/resource bounds, not provide population evidence or authorize outcome-driven design changes.

## 4. Recommendation

**Recommend (d): retain all 20 A worlds, prospectively amend the horizon to 1.5M, and retain the complete law panels and controls under the existing $60 cap.** Its component forecast is $46.10; it is the only listed option that fits. Preserve the nominal $13.90 headroom for repaired lifecycle accounting, fixed-input validation, and—if separately authorized—the ≤$1 hardware comparison. Do not spend that headroom by assuming a hardware speedup or cheap assays.

This deliberately prioritizes the primary route question and accepts a higher risk of **evolved law untested**. Report that risk in the amendment, withdraw the B and original late-horizon claims, and keep law support conditional on the actual eligible evolved panel and independent-world count. The historical evidence favors retaining 3M scientifically, but A-only 3M costs $76.03 and is not available under the present authorization.

Before any release: resolve B2–B5, validate the revised L2 controls and classifier boundaries, integrate the physical spend watchdog/metering, prospectively approve the changed scientific estimands, and refreeze the code, registration, cohort, schemas, forecast, and review identities. If the repaired complete forecast no longer fits $60, remain stopped and return a new proposal. This report changes no authorization and creates no launch receipt.
