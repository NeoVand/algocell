# S2 independent implementation review — 2026-10-12

Reviewer: a separate agent pass, assigned only this report. No implementation
files were edited by the reviewer. No GPU, Modal, soup stepping or population
observations were performed by the reviewer. The approved draft was read in
full, together with the additional L2 specification.

Status: **independent code review completed; STOP before launch**. The fixed
timing gate completed, but the complete unchanged design projects
**$138.721091 against the $60 hard cap**. Implementation findings have been
fixed and inspected. The approved L2 design defect in finding 1 also remains
unresolved. No population world or population pilot was dispatched.

## Blocking findings

1. **The approved L2 prediction is false for a permitted nonzero source start.**
   At L=32, d=16, use the 16-byte periodic seed beginning
   `BE 0C 00 BF 1C 00 49 F3 A4`, followed by seven arbitrary bytes. This sets
   SI=12, DI=28, CX=FFFF and runs forward REP MOVSB. Its kernel reads s+k and
   writes s+d+k with s=12, d=16. The baseline is periodic, the kernel covers B,
   there are no original-partner reads, and the baseline offspring resets into
   the same family. Its instruction footprint is C={0,...,8}. Its causal seed
   is original coordinates 12...27, however: original primary coordinates
   9...11 are erased, and originals 25...27 are retained. The approved claim
   that free coordinates are primary 0...d-1 minus C, with all originals d...L-1
   erased, does not follow. A CPU-generated ordered trace reproduces the
   extractor's acceptance and source map [16,...,27,12,...,15] repeated twice;
   the baseline output equals the parent, while the assay's canonical source
   check rejects it. The implementation must not silently change the approved
   theorem, rotate the seed, exclude this candidate based on retention, or call
   that baseline rejection a scientific discovery. A source-origin hypothesis
   or a shifted-seed law requires an explicit design resolution. This is an
   approved-design defect, independently of an implementation defect.

2. **Partial-register writes discarded preserved-byte dependencies.** The
   initial dependency transfer replaced all word-register taint for a byte MOV.
   Example at L=d=32: `BF 20 00 8B 0E 1E 00 88 C1 F3 A4`, zero fill, byte31=1.
   `MOV CX,[30]; MOV CL,AL` preserves payload-derived CH. REP's baseline count
   is 256, while changing byte31 to zero makes it zero. The initial extractor
   omitted that dependence. The implementing agent has changed to separate
   low/high-byte dependencies; a reviewer CPU synthetic trace now includes
   coordinate31 in C. Final regression verification remains required.

3. **Modulo-reducing control origins concealed erased-coordinate influence.**
   Initial C construction reduced fetched/data-control roots modulo d. A fetch
   or condition/address/count read rooted in original position >=d can then be
   counted as fixed primary core even though an erased-position intervention
   changes it before overwrite. The implementing agent now rejects these
   original erased-coordinate code/control dependencies. This guard is
   necessary for the primary-seed theorem; it does not repair finding 1.

4. **World settlements exceeded reservations and hard timeouts exceeded
   liabilities.** Initial projection reserved deadline*RATE while the remote
   world returned (deadline+120)*RATE. The original remote hard timeout was
   12 hours despite smaller planned allocations; cooperative checks cannot
   interrupt a hung compilation, GPU call or storage call. The dispatcher now
   uses per-job Modal `with_options(timeout=...)`. Verify reservations include
   the same termination/startup allowance and scoring/control allocations use
   that binding too.

5. **The original dispatcher did not dispatch scoring, extraction, laws or
   required controls through the spend guard.** The implementing agent is
   adding these stages. The later eligibility commit must be attested by a
   host-side comparison with the exact committed local eligibility files,
   rather than accepting a nonempty commit string at the remote function.
   Jobs must remain exclusively reserved and failed attempts retained, with
   automatic retries disabled.

6. **Constructed controls did not preserve all original intervention streams.**
   Initially `assay_case(control=...)` replaced serial partner streams only;
   joint assignments still came from new S2 streams. Required controls need
   original census single/joint assignments and original serial streams, and
   all ten L/d/operation combinations at budget128 must complete. Certificate
   checks alone are not that full positive-control assay.

7. **Fixed timing omitted actual runner submission overhead.** The original
   benchmark submitted 256 steps with one synchronization and reused one pair
   of parameter buffers; the runner uses at most50 steps, updates parameters
   per step in a ring, and synchronously reads the active-pair tally each
   submission. The benchmark must account for those actual costs while
   restoring soup, pairs and masks before every dispatch.

8. **Frozen projection and remote archive paths require binding.** The CLI's
   projection argument was independent of the immutable freeze manifest.
   Alternative deadlines/liabilities must be rejected. Remote images initially
   included only runtime Python files, algocell_exp and prelaunch artifacts,
   while wrappers read every path in freeze.files. Any registration, draft,
   tests or census contracts included in that manifest must be packaged too,
   or a separately frozen runtime identity list must be used consistently.

## Protocol observations

The chronological scorer scans every scheduled sample and keeps R0/R1,
occurrence, persistence and the route kill separate. Its culture threshold,
share gate, genotype tie rule and 11-sample persistence interval match the
approved draft. Unassessable qualifying top tapes mark the world unscoreable.
Random-cell multiplicities remain in the 64-cell denominator, while exact
genotypes are deduplicated for execution. Mechanisms come from the frozen
ordered full-address census classifier; numeric-PC-decrease heuristics are not
used. Full uninstrumented and diagnostic lane records are compared.

L1 uses the actual evolved baseline tape, checks the six-byte core and declared
displacement, and requires the constructed ordered-copy geometry. Its source
start is zero. The L2 dependency verifier deliberately leaves unsupported
opcodes unresolved; that admitted subset and any unresolved evolved
certificates must be disclosed rather than asserted to cover arbitrary 8086
loops.

The assay stages preserve mismatches and stop the affected case for saved-input
replays. A static source map is used for successful rows; its validity depends
on the independent noninterference certificate. The nonzero-source flaw shows
why an arbitrary constant displacement cannot substitute for proving the
registered source map.

## Final closure

Findings 2 through 8 are closed by final source inspection:

- Byte-granular taint preserves CH through a CL write, and INC/DEC conservatively
  propagate carries across word dependencies. Retained REP decode is preserved
  through continuations rather than lost on the first empty fetch group.
- Original erased-coordinate code/control dependencies are rejected before any
  retention assay. The primary core footprint is no longer formed by reducing
  those original roots modulo d.
- World and scoring/control reservations include120 seconds beyond the
  enforced function allocation. The dispatcher binds Modal hard timeouts to
  the individual planned allocation; cooperative ticks also cover extraction.
  Reservations reject nonfinite and negative amounts.
- All world, score, extraction, law and constructed-control stages dispatch
  through the append-only spend guard. Exact planned liabilities remain
  reserved for unstarted work. Eligibility data must match the released
  current Git commit and exact local committed files before assay dispatch.
- The ten constructed controls use census single/joint assignments and census
  serial partner streams. The complete control runner uses budget128 only.
- Benchmark submissions now reproduce the runner's50-step cap, parameter
  ring writes and synchronous active-pair tally reads, while restoring fixed
  soup, pairs and active masks before every individual dispatch. The final
  benchmark measures homogeneous named census witnesses, every constructed
  REP geometry/operation, a constructed non-REP loop, and the frozen mixture;
  the forecast uses the slowest repetition across these panels. Complete
  largest-d REP and non-REP law workloads are timed separately, with assay
  logging/flush costs, and the forecast uses the slower measured assay rate.
  Every panel's pair memory and complete lane records must match the direct
  machine executor before timing. The inherited soup readback's `(n,2,L)`
  shape is normalized to `(n,2L)` for that comparison. Remote NumPy/RNG fixtures
  must match the frozen fixtures before the benchmark starts, and actual
  Python/platform/NumPy/wgpu/Brotli versions are recorded in its result.
- The dispatcher rejects projections different from the frozen allocation
  plan. The remote image conditionally packages every additional path from
  the full freeze manifest; world, scorer and control wrappers verify those
  content identities.

A follow-up finding in the new law report was also fixed: an arm verdict now
requires20 complete scoreable records with the registered seed set. A partial
arm cannot report support. The L2-without-L1 cap-exclusion denominator removes
the excluded L1 cases consistently. Unresolved extraction certificates suspend
support rather than disappearing from the arm verdict.

The last closure pass also checked two assay fixes. Machine identity is compared
after JSON normalization, so a committed certificate's segment lists match
the runtime description's tuples without weakening its hashes/reset checks.
Mismatch diagnosis uses `baseline=False` only when replaying the deliberately
primary-only intervention: the baseline extraction still requires the actual
sampled parent to be d-periodic. A nonperiodic primary-only mutant is therefore
not rejected merely because the intervention followed the registered protocol.
Its copy geometry, execution identity, ancestry and expected output still have
to be diagnosed. Local GPU calls now record burst/pause durations and use a
finally block to preserve the equal pause after a raised exception.

The reviewer independently ran the initial CPU-only S2 protocol subset:
**12 passed, 8 deselected, 0.22 seconds**, then the expanded final CPU subset:
**19 passed, 8 deselected, 0.34 seconds**. These checks covered RNG fixtures, sample
cadence and ties, full-horizon/persistence boundaries, route cutoffs, candidate
selection/multiplicity, spend-liability/hash-chain checks, frozen fixed inputs,
known payload/core/source mismatches, overflow handling, snapshot identity,
and the archived ten-world DZ endpoint control. The expanded subset additionally
checked incomplete-arm law verdicts, all three injected pipeline mismatch
positions, nonfinite spend rejection without ledger mutation, and complete/
incomplete synthetic record validation. The implementation owner reported an
earlier full-suite result of **25 passed**, followed by the final combined
`tests/test_s8086.py` plus `tests/test_run6502.py` result: **38 passed in
34.46 seconds**, with one existing Modal-local mock warning. That full suite
completed all ten constructed law controls, totaling **1,185,792 encounters**;
exact census replay for both phases and budgets; the byte-count dependency and
nonzero-source counterexamples; and fixed pair-bridge/active-tally memory and
complete-record identity checks. The WGSL observer's reserved identifier
`active` was changed to `active_mask` before the successful validation. These
owner GPU results and the approved Modal benchmark are separate freeze
artifacts, not reviewer compute. No further implementation blocker was found
in the final separate pass.

Finding 1 is **not closed**. The implementation intentionally preserves the
approved L2 eligibility and prediction literally and adds a fixed-input
nonzero-source counterexample regression. It must be disclosed as a known
prelaunch design defect, never as a newly observed evolved-law failure. A
source-start restriction or shifted causal-seed prediction would be a design
amendment. This code review does not authorize population dispatch, and the
coordinator should resolve that scientific defect before approving any launch.
L1 is unaffected by that counterexample because its source start is zero.

## Remote import correction and manual preparation recovery

The first approved Modal preparation attempt failed during remote module
import, before benchmark function execution: the automatically mounted module
at `/root/modal_s8086.py` could not import `s8086.py` under
`/root/experiments`. The owner cancelled the call and explicitly stopped app
`ap-3yv8gfnaUhIahlAjZ7ZbF9`. The original call, fixed-input hash, failed-attempt
report and billing query are retained. The failed-attempt report explicitly
discloses Modal's repeated container import attempts despite function
`retries=0`; none executed the benchmark body. It records zero population
calls. The initial billing query reports zero metered dollars with billing-lag
qualification, and the full original $0.50 liability remains reserved.

The reviewer inspected the narrow correction: both wrappers add
`/root/experiments` to `sys.path` before top-level experiment imports, and the
Modal image sets the same `PYTHONPATH`. No fixed memories, cohort, endpoints,
law eligibility or intervention streams changed.

One explicitly selected manual recovery uses the distinct job
`preparation-benchmark-recovery`. An append-only `plan_extension` preserves
the original $0.50 failed liability, adds $0.50 recovery liability, and retains
$0.50 overhead: complete preparation liabilities are **$1.50 against the $2
preparation cap**. The guard reconstructs that extended plan from the ledger,
rejects reused job IDs and extensions beyond the cap, and rejects nonfinite or
negative extension amounts. The recovery entrypoint still has no automatic
retry. The eventual full-design preparation forecast conservatively includes
the failed $0.50 allocation as well as the recovery bound.

Independent CPU checks of this change: **3 passed, 25 deselected, 0.09 seconds**.
An additional reviewer CPU check injected NaN, infinity and a negative plan
extension and verified that each was rejected without appending any ledger
event. No reviewer cloud or GPU execution was performed. The code change and
single manual recovery introduce no new implementation blocker; the earlier
L2 scientific design blocker remains unchanged.

## Diagnostic backend correction and final bounded preparation recovery

The import recovery entered the benchmark but failed while NVIDIA Vulkan
compiled the census's 8192-event diagnostic shader, with
`GPUValidationError: Parent device is lost`. It produced no throughput timings
or population observations. The failed call and second failed-attempt/billing
artifacts are retained under app `ap-sCeRhIXHr8Hn8g5NSjXSpy`; the archived
billing query currently reports **$0.00354413**, with billing-lag qualification.
Its full $0.50 liability remains reserved alongside the first failed $0.50.

The explicit `S2_TRACE_BACKEND=software-vulkan` setting changes only the
diagnostic execution backend. `diagnostic_for` requests a fallback adapter
and asserts that it reports both CPU adapter type and Vulkan backend before
requesting the device. The Modal image installs `mesa-vulkan-drivers`. It runs
the unchanged `DiagnosticMachine(events=8192)` implementation and WGSL on that
device. Route scoring, eligibility extraction and mismatch diagnosis still
compare every output-memory byte and every complete original lane-record word
against the uninstrumented discrete-GPU executor. A mismatch suspends the
measurement. World, culture and uninstrumented law executions continue using
the shared GPU selector, which rejects CPU/unknown adapters.

The benchmark records both adapter identities, includes software diagnostic
compilation in startup measurements, and includes its execution/classification
cost in culture/route and extraction timings. No fixed memory, random stream,
biological design, encounter budget or trace-event cap changed. Unknown
nonempty `S2_TRACE_BACKEND` settings now raise before any adapter request; the
reviewer independently verified that rejection with a CPU-only check.

One final explicitly selected `--trace-recovery` creates a distinct ledger
job. The two retained failed liabilities total $1.00; its reservation is $0.50,
with $0.50 overhead. Complete preparation liabilities therefore equal **$2.00,
the approved preparation cap**. A reviewer CPU check reproduced this exact
total and verified that even a further $0.000001 extension is refused. This
does not enable automatic retries or discard prior attempts. The complete
design projection includes both failed $0.50 allocations conservatively.

The owner reports the expanded full suite passed **39 tests in 34.54 seconds**.
The reviewer also inspected the final reporting additions and independently
checked that 64 duplicate sampled-cell genotypes in each of 20 synthetic
records count as 1,280 sampled/culture-positive/eligible cells while selecting
only 20 distinct genotype-world cases. The assay summary now states coordinate,
alternative, assignment, partner and generation denominators explicitly.
There is no new implementation blocker in this pass. The known L2 theorem
scope defect still requires resolution before scientific launch approval, and
no population dispatch has occurred.

## Prospective storage-backed observer: feasibility review only

At the owner's request, the reviewer inspected a possible technical remedy if
the explicit software diagnostic compilation cannot complete. No observer
implementation or compute was performed in this pass.

The remedy is feasible while preserving the five-word event format and 8192
event cap. In both vendored WGSL exports, `main` defines local index
`i=zil_gid.x` and stores the lane record at `i*ZIL_RECORD_WORDS`. It passes
`zil_params.lane_base+i` to `zil_begin_at`/`zil_resume`; this is a global logical
lane identity, not the output-buffer row. `ZMachine.run` changes lane_base for
each host chunk. Using global lane times stride for direct trace writes would
therefore be an out-of-bounds/cross-lane defect.

A correct private record-base scalar can be set during `zil_reset_lane` as
`(lane-zil_params.lane_base)*ZIL_RECORD_WORDS`, before any event. The original
record prefix is 30 words at L16 and 32 at L32. With the unchanged six-word
diagnostic summary plus 5×8192 event words, extended strides are 40996 and 40998
words respectively. Direct event word j belongs at
`record_base+old+6+5*census_n+j`, for j=0...4, under the unchanged
`census_n<8192` check. Only the large private `census_log` array and its final
copy loop would be replaced. Event hooks, `census_written`, summary counters,
step/address/value fields, original-partner flag timing and the unconditional
event-counter increment must remain identical. Each invocation owns a
disjoint appended-record range, so no atomic operation or barrier is needed.
The base memory buffer and original lane-record words must never be changed.

Required validation before accepting such an implementation:

- On a backend that compiles both variants, compare the complete diagnostic
  records and valid ordered traces bit-for-bit, in addition to raw-memory and
  complete original-record equality against the uninstrumented executor.
- Replay the four census words, both phases, both lengths/budgets and all REP
  and non-REP certificate controls; retain classifier/certificate parity.
- Test a nonzero lane_base with forced host chunks; mixed per-lane budgets
  including zero; early halts; retained REP and resume; summary-only events=0;
  small-cap overflow with the total count continuing and logs becoming unknown
  only when the total exceeds the cap.
- Preserve unused trace-tail behavior when a prior record is reused/resumed;
  valid length is the recorded count, not a newly invented clearing rule.
- Assert the derived trace writes are confined to the appended range; archive
  new observer/source hashes and disclose the implementation change. Any
  successful timing must measure the actual final observer and reproject all
  assay costs.

This assessment is not validation of an implemented replacement and does not
authorize another cloud allocation. The currently reserved preparation
liabilities equal $2; any further cloud attempt requires a separately recorded,
prospective reservation that still fits the existing approved cap. It does
not alter the unresolved L2 scientific design defect.

## Storage-backed observer implementation closure and stopped-app settlements

The owner subsequently implemented optional `storage_trace=True` in the
existing `DiagnosticMachine`. The reviewer inspected that implementation:
local record-base initialization precedes events; direct stores remain within
the appended trace region; event fields, order, counters and overflow policy
are unchanged; the large private array and final trace-copy loop are absent
only in the optional storage mode. No classifier rule changed. The final
Modal setting explicitly selects `S2_TRACE_BACKEND=storage-vulkan`, using the
raw machine's discrete-GPU device for the equivalent storage-backed observer.
Every measured trace retains raw-memory/full-original-record verification.

An independent CPU-only generation check compared default observer WGSL and
manifests against pre-storage commit `92f8a24`: they are identical at L16/L32
for event caps 0, 1, 5 and 8192. The check also verified removal of private trace
references and exact output event ranges: L16 words 36...40995 in stride 40996,
and L32 words 38...40997 in stride 40998. Original record words and the six
summary words lie outside those ranges.

The owner reports the full expanded suite passed **40 tests in 64.14 seconds**.
The new storage-equivalence test compares complete old/new diagnostic records
and memory for all named census words in both phases, lengths and budgets;
REP/non-REP controls; chunked dispatches with nonzero logical lane bases;
mixed 0/1/3/128/512 budgets; small-cap/summary-only overflow behavior; and
resumed REP. The reviewer noted missing explicit memory comparisons in the
REP/resume portions; those assertions were added without additional
dispatches and inspected. Successful full-record equality also establishes
bit-exact stored event tuples and ordering. The reviewer did not run GPU tests.

The software-private-array attempt reached its 450-second timeout without
timing results; its failed call and stopped-app artifact are retained. All
three failed apps are now stopped, and the ledger appends settlements rather
than removing or overwriting their reservations and provenance. The recorded
conservative bounds are:

| Stopped attempt | Whole-app resource bound, USD |
|---|---:|
| Import failure | 0.1361584456 |
| NVIDIA private-trace compilation failure | 0.3296596769 |
| Software private-trace timeout | 0.2319970068 |
| Total | **0.6978151293** |

Each bound uses the recorded whole elapsed app time plus 120 seconds and $0.05
image allowance; it is not presented as an invoice. The reviewer independently
validated the append-only ledger hash chain, the sum of settlements against
`failed_resource_bounds.json`, and the prospective storage-attempt liabilities:
**$0.6978151293 + $0.50 new reservation + $0.50 overhead = $1.6978151293**, below
the approved $2 cap. Lowering the liability after confirmed app stop is
supported by the archived bounds; it does not erase any technical failure.
The complete design projection includes all stopped-attempt bounds plus the
new attempt's bound. No fixed memories, event capacity or scientific protocol
changed.

No remaining storage-implementation blocker was found. The subsequent timing
gate and final cost stop are recorded below; the known L2 scope defect remains
a separate launch blocker.

## Final prelaunch closure and cost stop

The reviewer inspected the final bookkeeping and proof corrections:

- Per-world scores and eligibility ledgers use deterministic gzip JSON with
  exclusive creation. Local committed eligibility paths, remote stage paths,
  receipt hashes and assay readers consistently use `.json.gz`. The receipt
  attests the exact compressed bytes. Law reporting always emits L1, L2 and
  L2-without-L1 separately for each arm, with the registered candidate/cell/
  case denominators; there is no secondary rescue of L1.
- Flag dependencies are separated into CF/PF/AF/ZF/SF/OF. Word INC/DEC preserve
  CF, propagate low-byte carry/borrow into the high byte, and assign only the
  relevant dependency sets to updated flags. Each Jcc consults its actual
  flags; LOOPZ/LOOPNZ consult ZF. A fixed semantic JNC loop therefore does not
  promote payload merely because INC changed unrelated arithmetic flags. The
  reviewer found its synthetic fixture's branch byte FD targeted INC rather
  than LODSW; it was corrected to FB (next IP8 to target IP3) and its CPU
  regression passed. The measured REP/LOOP controls have unchanged geometry,
  streams and source behavior under this transfer correction.
- An unknown qualifying culture blocks declaration of a later first
  replicator/confinement endpoint until diagnosis. Unscoreable worlds have an
  explicit denominator and are excluded from D and the Kaplan–Meier biological
  censoring population. Complete-arm requirements still prevent a provisional
  partial arm from deciding registered route thresholds.
- `Unresolved` trace/certificate errors in mismatch diagnosis produce
  incomplete/invalid status, rather than a scientific application failure.
  Saved-input replay, full original-record/memory identity and zero-tolerance
  rules remain in place.

The reviewer independently reran the final CPU-only S2 plus runner subset:
**35 passed, 9 previously covered GPU cases deselected, 0.49 seconds**, with
the existing Modal-local mock warning. The owner previously ran the full
40-case suite and then the focused storage memory assertions; the final
combined coverage is **44 distinct cases**, with later added CPU cases
verified after the corrections. No reviewer GPU or Modal execution occurred,
and no further GPU/Modal compute occurred after the over-cap timing gate.

The successful fixed-input L4 benchmark is archived with the unchanged input
manifest and raw/census/full-record observer parity checks. The reviewer
independently checked its records and reproduced the final projection from
the benchmark, preparation bounds and output-size metadata:

| Measurement | Value |
|---|---:|
| L16 slowest fixed world throughput | 350.2020336 world-steps/s |
| L32 slowest fixed world throughput | 425.4187037 world-steps/s |
| Successful benchmark function body | 79.7481493 seconds |
| App-metered preparation across all four attempts | $0.13974218 |
| Conservative whole-app preparation bound | $0.79990382 |
| Preparation bound plus overhead reserve | $1.29990382, below $2 |

The successful app contributes $0.02075924 of the currently metered total.
All four apps are stopped. Metered figures are archived observations with
billing-lag qualification; conservative bounds remain the spend guard's
liability basis. The reviewer validated the final append-only ledger hash
chain, all settled-bound sums and all app-metered sums against the summary.

The complete forecast includes all 40 worlds at 3M steps, every culture/route/
extraction/law workload at its registered ceiling, all ten controls, startup
and per-job termination allowance. It adds a 1290-case diagnosis reserve
(1280 possible L1/L2 genotype-world cases plus ten controls), and $5 for
storage/export/termination. Output volume uses twice the measured worst
gzip bytes per encounter from the full passing controls, plus 2 GiB overhead,
one paid month and one export: approximately 31.053 GiB. Planned longest world
function allocation is 11982 seconds, with the additional 120 seconds priced
per job. The sum of all job liabilities plus the storage reserve reproduces
**$138.72109138536507**, and `launch_fits` is false.

The design was not reduced or tuned to fit the budget. The runner/dispatcher
release checks reject this frozen projection under the $60 cap before
population work. There is no remaining implementation finding in this final
pass; the confirmed budget excess and known L2 theorem-scope defect require
the work to remain stopped. Freeze/registration assembly is permitted, but
this report supplies no launch approval.

Reviewed source identities (SHA-256, before final registration/benchmark
artifact assembly):

| Source | SHA-256 |
|---|---|
| s8086.py | `a213bede4bfe0a2fbdbb7f76102ca9ddd6a2f71741c3e341caaee5efcfcc071b` |
| run8086.py | `ff08fe041a2592c621c1f5fd49a4d185477677a16f9c8ee5ee429c78f991e585` |
| score8086.py | `b5c0325506588352a7fc5046a4eb3b99760c3e3d0cdc079e268d026d1733ab7f` |
| law8086.py | `0325bc086cd2d532c9de1466f8e69d8211ac0ee9e49c5f397ab9c28c0d895ce4` |
| dispatch8086.py | `7871d63cf49b1f583eee1d52da5d1925ea039bbfb06becc4f1d4f714075624fe` |
| benchmark8086.py | `e070002c562d42305585352867e4bc97671631285e2b64bcfd0a68610f7a8997` |
| modal_s8086.py | `04a74448f025b61104d3bf7793381ea0c9dd1db0ecf287f653530ec8569de5f2` |
| modal_score8086.py | `672372a0208dc39f66c5a791236ff46bcb48c881e1007e921c0c39ba853fc33d` |
| algocell_exp/census8086_trace.py | `d0c239b6258b6f4b793289304aeea190ae78f828dff4841945ecbc2aac8655de` |
