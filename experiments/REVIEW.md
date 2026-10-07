# Pipeline review — 2026-10-07

Trigger: two bugs reached GPU sweeps before being caught (the assay executor's fixed 40-byte memory, which made every post hoc assay for L ≥ 25 garbage until per-L executors were exported; and `stop_share: -1` being treated as a positive share, which stopped all 420 Stage C runs after 2,500–5,000 steps). The owner asked for a full review before any further spend. Four independent reviews were run (runner + Modal infrastructure; assay + ISA + metrics; post hoc analysis + figures; experimental design), each read every line in scope and verified claims by executing code against the Stage A/B data. This file consolidates their findings, what was changed, and how each change was verified. Raw simulation data (Stages A, B; 1,270 runs) is unaffected by every finding below unless stated.

Severity: BLOCKER = a conclusion or a relaunch would be wrong; MAJOR = a reported number or a run could be wrong; MINOR = robustness/clarity.

## 1. Simulation runner and Modal infrastructure

| # | Sev | Finding | Fix | Verified |
|---|---|---|---|---|
| R1 | MAJOR | `exemplars()` labels mechanisms without the run's suppression set, so a suppressed instruction (a NOP on the GPU) is counted as a mechanism; changes the label in 10/277 Stage A and 9/336 Stage B runs with a stored first-emergent tape | `exemplars(..., suppress=soup.sets)` in `run.py`; analysis recomputes mechanisms from stored tapes + suppression (no re-run needed) | done: `exemplars(..., suppress)`; analysis recomputes from stored tapes |
| R2 | MAJOR | A run killed mid-flight leaves a `soup_emergence` snapshot from a different trajectory next to the re-run's files (8 orphans in `runs/stageD`) | `run_condition` deletes `{stem}.*` before running; analysis loads the emergence snapshot only when `tq_10 ≥ 0`; summary written last and atomically | done: `batch.run_to_dir` removes `{stem}.*` first; orphans on the volume are removed by the relaunch itself |
| R3 | MAJOR | `run_condition.map` without `return_exceptions` and no retries: one failed container aborts the client, which disconnects the app and kills every in-flight run | `return_exceptions=True`, failures logged, `retries=Retries(max_retries=2)`, summaries flushed as they arrive | done; exercised by `preflight.py` (same code path) |
| R4 | MAJOR | `fetch.sh` swallowed download errors (`|| true`, stderr to /dev/null); `remaining.py` then re-runs (and overwrites) finished cells | both fail loudly; `remaining.py` refuses to run without a fetched directory and prints the stems | code changed |
| R5 | MINOR | Adapter choice ignored the backend; an OpenGL adapter of the same GPU could win on enumeration order | sort by (adapter type, backend) with Vulkan/Metal first; reject CPU and Unknown | code changed; all 1,702 existing summaries read "L40S (Vulkan, DiscreteGPU)" |
| R6 | MINOR | PLAN interaction counts wrong: 8192 pairs are drawn but the collision claim leaves ≈ 4.7k active (57%); ≈ 1.4×10⁹ interactions per 300k-step run, ≈ 1.4×10⁵ participations per cell (PLAN said "≈ 120") | PLAN text corrected | numpy re-simulation of the pairing kernel |
| R7 | MINOR | `mutate_soup` does a non-atomic byte read-modify-write on u32 words, so two mutations in one word in the same step can drop one: ≈ 0.3% of mutations at L = 16, k = 4; up to ≈ 5% at L = 4, k = 2 | documented as an effective-rate correction in the methods; a per-word atomic fix changes the shader and is scheduled with the Zilion upgrades | expected loss computed analytically |
| R8 | MINOR | Modal image used unpinned `wgpu>=0.32,<0.40` | pinned to the versions the tests run against (wgpu 0.32.0, numpy 2.5.3, brotli 1.1.0) | code changed |
| R9 | MINOR | CLI `--out` opened in append mode (duplicate condition/summary lines on re-run) | write mode | code changed |
| R10 | NOTE | `mutation_rate` field is mutated bytes per pair slot, not per byte or per cell | kept for compatibility; `mutation_per_cell_per_step` and `mutation_per_byte_per_step` added | code changed |
| R11 | NOTE | Stage D stems equal Stage B's L = 9 stems; only `--batch` separates them | the launcher refuses a `--batch` that does not match the condition file name (`--force-batch` to override) | done |
| R12 | NOTE | `random_tapes` RNG seeded by `seed·1_000_003 + step` (collision-free only because horizon < 1,000,003) | `default_rng([seed, step])` | code changed |
| R13 | NOTE | FNV-1a 32-bit species hash: P(any collision among 20,000 distinct tapes) ≈ 4.5% per sample; shifts `unique` by ±1 | documented | — |
| R14 | NOTE | "Re-running a seed" is not a replication: four local seed-1 runs diverged (q_share at step 1000: 0.084/0.074/0.076/0.106, different dominant tapes) because of atomic claim order and the RMW race | PLAN text corrected; each run is one sample, as designed | measured |

Verified correct by review 1 (not changed): params uniform layout and mask bit order (bit-exact against the 13 TypeScript golden vectors); params-ring ordering (no CPU/GPU race); SplitMix64 stream and soup initialisation bit-identical to the browser; mutation count 8192 ≫ k at every L (per-byte rate ∝ 1/L); one Z80 step = one instruction, each LDIR iteration one step, HALT forfeits the budget; sampling at k·sample_every with the last chunk shortened, `steps_run` = last sampled step; early stop after the fix (−1, 0, None never stop; 1e-6 stops at 500 + 4·500); 8 random tapes per sample (≤ 2.7 MB per run); snapshots present iff `tq_10 ≥ 0` in all 1,270 runs; Modal timeout ≥ 40× the longest run; condition files (630/640/420/160, no duplicates, every suppression string resolves, Stage D equals PLAN field-for-field); `remaining.py` stems equal `run_stem` for all 1,850 conditions; exported shaders current (sha256 prefixes match `meta.json`, zilion 0.2.0).

## 2. Post hoc analysis, statistics and figures

| # | Sev | Finding | Fix | Verified |
|---|---|---|---|---|
| A1 | BLOCKER | `zoo.py` "final" designs use the emergence-time tape (the `where` tape) instead of the in-situ winner; 283/346 Stage B "final" rows are the wrong tape | `assay_batch` stores `final_insitu_tape`; zoo uses it | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A2 | BLOCKER | FINDINGS §1 L = 16 row said 10/10 and 10/10 for stack-writes; `cells.csv` says 9/10 and 8/10 (the 10/10 was the end-state in-situ measure) | FINDINGS regenerated from code (see §4) | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A3 | BLOCKER | Two "size axis" figures plot different quantities (tq_10 vs t_rep fraction) under the same label; they disagree in 18/64 cells | `analyze.py` version removed; one figure, measure in the title | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A4 | BLOCKER | Stage B Kaplan–Meier figures: 32 curves per axes, legend over 60% of the plot, colours repeat | facet by L, colour = ablation (fixed palette) | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A5 | MAJOR | Mechanism-class column anchored to the first 2% exact-share crossing and labelled without suppression | recompute for the `t_rep` tape under the condition's suppression; label the t_02 one as such | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A6 | MAJOR | `final_period` computed on the final top-1 exact genotype, often sterile debris (`01 01 01 …`, period 1, in 11/40 L = 25 runs); `minimal_period` returns L on a single defect | period on the in-situ winning tape; tolerant period (≥ 90% of positions match) with the match fraction stored | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A7 | MAJOR | Medians over emerged seeds reported as "median t_rep"; KM medians differ (stack-writes L = 49/128: 21,500 vs 16,250) or are not reached (none L = 9) | KM median (NR when not reached) in every table; conditional medians only with "(among k/n)" | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A8 | MAJOR | `is_replicator/score/gen2/copies75` describe the emergence-time tape but were summed as `final_rep_rnd`; FINDINGS §5 compared a step-1,500 exemplar with the final in-situ assay | columns renamed `em_*` / `final_*`; final tape assayed against random partners too | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A9 | MAJOR | FINDINGS §3 zero-flood numbers were L = 4 and block-copy values attributed to L = 9 (`00 39` is a 4-byte tape; 7.4 bits is stack-writes at L = 4) | regenerated from snapshots with L, step and statistic stated | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A10 | MAJOR | FINDINGS §2 functional fractions wrong in both ranges; in-situ fraction is 0 by construction in saturated soups | report random-partner functional fraction as primary, in-situ as secondary with the caveat | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A11 | MAJOR | FINDINGS §2 succession sentence wrong (takeover n = 6/10 at L = 100/128; "later invasions" at L = 81/100 were LDIR from the start) | regenerated | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A12 | MAJOR | `final_family` is evaluated at the early-stop step (5k–20k in 144/640 Stage B runs) without saying so | `last_step` column in every succession table | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A13 | MAJOR | `report.py` verdicts hard-wired to L = 16 (empty for Stage B); H6 not implemented; H1 lacks the CI-overlap condition; H4/H5 lack the sign test | parameterised; H6 added; tests added | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A14 | MAJOR | Colour → ablation mapping differs in every figure family (and solid/dotted of the same ablation got different colours) | one `figstyle.COLOR` dict (Okabe–Ito, control black) used everywhere | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A15 | MAJOR | `size_axis/*.png`: overlapping clipped titles, legends on data, no-copy's empty Wilson bars dominating, n = 1 medians plotted like n = 10 | rebuilt to the spec in §5 | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A16 | MINOR | `t_rep` hit's faithfulness, score, rank and share were dropped | stored with `trep_` prefix | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A17 | MINOR | `where` set before checking that the emergence exemplars exist (latent) | fixed | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A18 | MINOR | missing jsonl → −1 (indistinguishable from censored); truncated last line aborts the batch | NaN + warning; partial lines skipped | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A19 | MINOR | zoo signatures: ld8-imm registers not generalised; multi-instruction units not collapsed (522-char signatures); phase splits one design in two | period-based signature (one period, "×n"), canonical rotation | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A20 | MINOR | `succession.py` threshold is headroom-normalised excess, PLAN says "30% excess"; incomplete invasions dropped from the duration median | PLAN wording aligned; incomplete invasions counted as censored | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A21 | MINOR | `publish_results.sh` skipped `size_axis/` and hid copy failures | fixed | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A22 | MINOR | KM: unstable tie order (censoring removed before a tied event), x-axis starts at 0 on a log axis | events first; axis from the first sample | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A23 | MINOR | further FINDINGS discrepancies (none L = 16/512 median 1,000 not 500; period 32 at L = 64 only in t_rep; aperiodic LDIR 20/5/5/0% at L = 49/64/81/100; two-sided Fisher not stated) | regenerated | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A24 | MINOR | Stage B "heatmaps" are 1×2 cells; values as "0.4" not "4/10"; no uncertainty | bar panels with Wilson intervals | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A25 | NOTE | the unfaithful CALL chain appears as "heritable at end" in one figure | faithfulness filter applied | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A26 | NOTE | `final_rep_fraction` from 16 cells (SE ≈ 0.12) | 32 cells, random-partner and in-situ variants, Wilson CI | done (2026-10-07 rewrite; Stage A/B re-analysed) |
| A27 | NOTE | by the pre-registered `tq_10` the L = 9 reverse ablation points the other way (tq_10 fires on the zero flood) | FINDINGS states the finding exists only under the assay measure; Stage D pre-registers the assay as the outcome | done (2026-10-07 rewrite; Stage A/B re-analysed) |

Verified correct by review 3: Wilson intervals; `t_rep` scan (time order, 0.5% share, first heritable hit, complete cache key); sentinel handling; the early stop never censored `t_rep` in Stages A/B; in-situ snapshot sizing per L; `succession.py` recomputed exactly from the jsonl for four runs; zoo disassembles under the right suppression; Stage A L = 16 cells pulled at matching k and steps.

## 3. Assay, ISA and metrics

The executor path is correct at every tape length: params indices, io layout and write-back verified on the GPU with register readback at all nine L (SP = spInit(2L): 0xFFEF at 18, 0xFFDB at 50, …; first PUSH lands at mem[2L−2], mem[2L−3]); hook widths match zilion 0.2.0 (base 1-byte NOP, CB/ED 2-byte, DDCB 4-byte, DD+base 2-byte with operands falling through); mask bit order identical in Python, TypeScript and WGSL (13 golden + 21 extra vectors); exported shaders byte-identical to a fresh export. Known copiers score 1.00/1.00 at every L (table in §7).

| # | Sev | Finding | Fix |
|---|---|---|---|
| M1 | MAJOR | mechanism labels ignored the suppression set (same as R1) | done |
| M2 | MAJOR | `gen2_score` is a two-generation *yield* (offspring the tape failed to convert are included), so the same mechanism crosses 0.3 or not depending on L and budget (`21 e3 21 e5`: 0.29; Load–Push at L = 9: 0.32) | `gen2_cond` (heritability given a copy) reported next to the pre-registered gen2; semantics documented in the module docstring |
| M3 | MAJOR | in-situ `_norm_gain` returned 0 ("sterile") when no partner had headroom; in soups ≥ 90% saturated the random-cell fraction was biased low | NaN with `n_informative`; partners re-drawn until ≥ 16 are informative |
| M4 | MAJOR | `q_share` is Hamming without shift while stack copiers write phase-shifted copies (under-count up to 3×; `tq_10`/early stop inherit it) | `q_shift_share` (best cyclic shift ≥ 0.75, 2,000-cell subsample) recorded per sample; pre-registered `q_share` kept |
| M5 | MAJOR | faithfulness needs a conjunction: the RST smear passes `offspring_within_q` alone (0.64), the CALL chain passes gen2 alone (0.49), generation 3 alone also fails (the s10 chain returns to phase) | faithful := gen2 ≥ 0.3 ∧ offspring_within_q ≥ 0.5, used everywhere |
| M6 | MINOR | no null for partner-induced similarity (all-zero tape up to 0.18 / 0.16 at L = 36–49, 512 steps) | documented in §7; threshold margin stated |
| M7 | MINOR | demo tapes `1e 20 ed b0` (DE = 32 ≡ 0: copies A onto itself) and `c4 5e ed b0` (CALL NZ) were not copiers | replaced by `1e 10 ed b0` and a tiled `1e 04 ed b0` |
| M8 | MINOR | `disassemble(wrap=…)` dead parameter; operands past the tape end truncated | parameter removed, behaviour documented |
| M9 | MINOR | census definitions undocumented and not L-fair (random baselines 0.1% → 47% for `c_push2`) | definitions written out; per-L baselines already used by succession.py |
| M10 | MINOR | `motif_share` is 0 for aperiodic dominant tapes | documented; not used for conclusions |
| M11 | MINOR | no executor/copier/suppression-width tests | `tests/test_pipeline.py` (40 tests) |
| M12 | NOTE | B-role blind spot: `gen2` is A-role only; a B→A-only copier is theoretical (it copies the partner over itself as A) | `self_preserved_as_B` added (vulnerability); roles documented |

## 4. Experimental design (referee report, adopted)

| # | Sev | Finding | Response |
|---|---|---|---|
| D1 | BLOCKER | Stage D re-used seeds 1–20: a seed fixes the initial soup and RNG stream, so seeds 1–10 were Stage B's own soups | seeds 1001–1020 (and 101–110 for C/E); test enforces it |
| D2 | BLOCKER | the `stack-writes` arm removes 46 opcodes of which 24 write nothing (POP, RET*, EX DE,HL, EXX, EX AF,AF') and are the LDIR zoo's pointer routes; H2 and the L = 9 reversal are confounded | `stack-write-only` (22) and `stack-read-only` (24) arms in C3 and D; `push` (4) and `call-rst-write` (17) in D |
| D3 | BLOCKER | emergence times at the 500-step sampling floor (25–36% exactly 500) | sampling every 50 steps to 5,000 in every C/D/E run |
| D4 | BLOCKER | Stage D predictions not recorded (zero fraction) or not discriminating (the flood is universal: 0.24–0.34 at the tq_10 crossing at every L) | byte histogram + zero fraction per sample; D2 reframed as a premise check; D3 defined as a ratio; D5 added |
| D5 | MAJOR | steps-per-byte confound visible: stack-writes at L = 81/100, 128 steps are heritable-but-unfaithful | faithful column everywhere; Stage E @steps8L arm |
| D6 | MAJOR | parity of L drives the early stop (even L stop at 2.5–5k, odd L run to 300k) so final-state comparisons mixed ages | fixed-step family columns (5k/50k/300k); C/D/E without stop; non-square L in E |
| D7 | MAJOR | per-byte mutation ∝ 1/L, lottery size ∝ L, steps/byte ∝ 1/L can each produce "flat above 16" | Stage E control arms @mubyte, @bytes, @steps8L |
| D8 | MAJOR | pre-registered primary tq_10 fails (floods); H3 refuted by its letter | stated in PLAN and REPORT; A/B classified exploratory, outcomes fixed for C–E |
| D9 | MAJOR | assay validity: yield vs heritability, saturation, cyclic matching, A-role only | §3 fixes; faithful co-primary; `gen2_cond`; vulnerability |
| D10 | MAJOR | conditional medians; no KM; paired design unused | KM medians + NR; seed-paired sign tests in report.py |
| D11 | MAJOR | hypotheses without effect sizes / MDE; H5 lacks the low-mutation side | restated in PLAN for C–E; k ≥ 8 deferred (cost) |
| D12 | MAJOR | multiplicity over 127 cells × outcomes | pre-registered family with Holm; atlas scan reported with counts; post hoc tables never carry verdicts |
| D13 | MAJOR | reproducibility: shader/ISA hashes, commit, versions, driver not recorded per run; non-determinism unmeasured | provenance in every record; E6 within-seed variance arm |
| D14 | MAJOR | novelty: incremental parts (stack-first, no-copy null) vs new parts (prefix-aware NOP ladder × budget × mutation; heritability/faithfulness vs occupancy; tiling; the reversal) | paper claim restated in RESEARCH_DIRECTIONS.md §5 pending C–E |

## 5. Figure specification (from review 3, adopted)
One `figstyle.py`: Okabe–Ito palette with the control black — none #000000, block-copy #E69F00, stack-writes #56B4E9, ld-mem #009E73, all-ld #CC79A7, no-copy #D55E00, rmw-only #999999 — and a separate muted set for census families; 8 pt body, 9 pt titles, 7 pt ticks; no top/right spines; legends outside the axes; PDF + SVG + 300-dpi PNG; 85 mm or 180 mm widths; every fraction with a Wilson 95% interval, every time with a KM interval or "NR"; solid = pre-registered measure, dashed = post hoc assay measure. Figures: F1 emergence atlas (Stage A, tq_10 and t_rep rows, "k/10" text); F2 KM small multiples (steps × k, colour = ablation, censor ticks); F3 size axis (fraction, KM median, final family composition); F4 L = 9 reverse ablation (KM, Fisher/log-rank, zero fraction over time); F5 tiling (tolerant period of the in-situ winner vs L, divisors filled; HOE by family).

## 6. Statistics policy (adopted)
Fractions: Wilson 95%; contrasts: exact Fisher (one-sided only where pre-registered), Boschloo when scipy is available; time-to-event: KM with censoring at `steps_run`, exact permutation log-rank for 10 vs 10 (all 184,756 relabellings enumerated); monotonicity: the pre-registered sign test on paired seeds; multiplicity: Holm within the pre-registered family, Benjamini–Hochberg q for the atlas scan with the number scanned stated; two tables per stage, "Pre-registered" and "Post hoc", the latter never carrying a verdict.

## 7. Calibration of the heritability assay (measured 2026-10-07, L = 16, 128 steps)
200 random tapes: gen2 mean −0.009, sd 0.009, max 0.027; score max 0.083. All-NOP tape: score 0.11, gen2 0.13 (random partners push zeroed registers). Sterile offspring `21 e3 ×8`: 0.22 / 0.05. Parent `LD HL,$E321 ; PUSH HL`: 0.63 / 0.29. RST smear: 0.62 / 0.09. Load–Push `01 c5 ×8`: 0.82 / 0.59. Tiled LDIR copiers: 1.00 / 1.00 at every L from 4 to 100 (the L = 9 unit is `DEC E ; LDIR`). Threshold 0.3 is > 10 sd above the random-tape null and 2.3× the worst null.
