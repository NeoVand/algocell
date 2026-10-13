# S2 B2–B5 repair report — 2026-10-12

**STOPPED.** The user is choosing a prospective cohort. No final registration,
freeze, launch receipt, population work, new Modal call, or paid computation was
performed. The original registration, freeze.json, projection.json, benchmark,
and preparation billing evidence remain unchanged. Source changes intentionally
invalidate the old freeze. Astra's uncommitted review was committed first,
unchanged, in `b21e0f7`.

B3 now records the start hook's actual `pc`, for ordinary instructions, prefixes,
far transfers, ring aliases and retained REP. The exact GPU boundary witness
`BF 10 00 B9 02 00 F3 A5 F4` reproduces the reviewed observer's starts
`[1,4,7,6]` and one-pass label, then verifies corrected starts `[0,3,6,6]` and
confined-loop label. Full original lane/memory parity holds for both observers.
Sequential/far/near/self-modifying/halt/resume/storage fixtures also pass.

B4 uses ordered per-byte read-event identities and values, including partial
register transfers. Arithmetic clears memory-copy identities; no membership in
read values establishes pairing. The GPU `MOV SP,[14]; PUSH SP` witness stores
`12 12` at pair positions 18/19: the reviewed classifier labels it memory-fed,
while the repaired classifier returns **other**. CPU regressions also reject
swapped source bytes and stale SP/SI/DI/CX/REP provenance. The implicit-update
audit covers SP push/pop/segment/call/return/interrupt helpers, SI/DI strings,
CX REP/LOOP, accumulator transforms, DX sign-extension/multiply/divide and
unknown/trap paths. Supported unaffected registers survive segment operations;
unsupported transformations poison provenance. Changed RMW values remain other;
address-paired unchanged RMW retains the existing contract.

The fresh Gate 2 audit reran **1,246 candidate settings / 21,182 encounters**
against the entire frozen 17-partner panel. Every original record and memory
matches raw execution; match, coverage, access and event-count columns are
unchanged. There are **1,476 changed labels**: 1,440 other→literal writer and 36
other→memory-fed one-pass copier. Every change is listed in the dated Correction
section of `experiments/results/census8086/CENSUS.md` and `correction_labels.csv`.
`B9 51` changes eight random-partner labels to literal writer; its zero label
stays literal. `FF 30` and `AD 50` change no labels. Two known trace overflows
remain other. Full corrected traces for every non-overflow encounter are
archived in bounded gzip parts; the large historical law serial table is now
gzipped with identical decompressed bytes.

**07 A5 is still an open memory-fed one-pass copier.** Its 17/17 mechanism labels
remain one-pass in every length/budget setting; its 16/16 random encounters
fetch B. The registered R2 predictions have not been altered.

B2 implements Astra §2's source-root/core/phase amendment prospectively.
Certificates keep raw K and the c≤d footprint, certify actual R and ordered r,
then separately record C_R, F, E, E\K, held E∩K, and physical phase zero. The
predicted child is x[r(q)], without alignment search or prefix repair. Mutants
hold K and C_R; F and E\K use the exact registered all-value/joint streams.
Serial reports carry the composed generation-zero ancestry, while diagnosis
keeps raw dependencies separate. Unsupported transfers remain unresolved;
reports retain unresolved counts and independent-world coverage, plus source
geometry and erasure denominators. Constructed success remains separate from
support on an evolved panel.

For L32/d16/s12: K=0…8, R=12…27, C_R=16…24, F={12,13,14,15,25,26,27},
E\K={9,10,11,28,29,30,31}, and nine held erased setup coordinates. The source
seed phase is 4 modulo 16; physical offspring entry remains zero. All **88,064**
encounters pass with zero mismatches: 1,792 single assignments and 512 joint
assignments against 17 census partners and through eight actual offspring
transfers, plus 1,792 erasure assignments against 17 partners. Their original
held setup ancestry is absent. All ten s=0 REP controls pass the unchanged
**1,185,792** encounter suite, with original streams, canonical source maps,
zero phase, and unchanged 2/10/26 free-coordinate counts. The non-REP and raw
byte-dependency certificates also pass.

B5 adds explicit 120s startup limits and zero idle/buffer containers to every
function. Each job uses a single app/container pool; a separate host process
is armed before spawn/get. All startup and infrastructure attempts share one
monotonic envelope; attempts never renew it. The watchdog polls lifecycle and
billing independently, cancels calls, stops the app with bounded RPCs, and
requires provider STOPPED plus an empty task list. A result never settles a live
app or allows the next job. New metering affects the same ledger transaction
and drives stop decisions. Failures retain reservations; overruns/unknown rate
bounds prevent new reservations and settlement pending reconciliation. A
host-side hash-chained terminal row records the original failure, app/call
identity and last acknowledged progress independently of remote GPU/storage.
Runner initialization is inside the failure handler and never requires another
GPU read to preserve its original error.

Mock tests exercise a blocked parent with an independent process, startup and
restart clocks, cancellation/termination, current-transaction metering, unknown
stops, live tasks despite STOPPED, terminal failures, and settlement refusal.
No Modal test was run. Provider availability and termination latency remain
operational assumptions: an unconfirmed overrun is explicitly unknown, cannot
be certified as physically bounded, and cannot count as completed biology.

Horizons and arm sets are registration parameters, propagated through runner,
score schema, occurrence, censoring, final-top/64-cell sampling, completeness,
route/law reports and dispatcher. Samples truncate the original schedule;
checkpoints are distinct original points through H plus final H. R2b uses
exactly 11 points H−250k…H at 25k spacing, including the 11-point subset of the
101 scheduled samples in the final 250k at H=1M. Shortened records cannot pass
the original 3M completeness check. Twenty-world integer cutoffs are unchanged.

| Option | Cohort | World | Score | Extract | Laws L1+L2 | Fixed | Complete | $60 headroom |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| (a) | A+B, 20 each × 3M | $114.318006 | $8.187607 | $5.653950 | $14.205675 | $6.241397 | **$148.606635** | −$88.606635 |
| (b) | A, 20 × 3M | $62.421020 | $4.035646 | $2.826975 | $5.507291 | $6.241397 | **$81.032328** | −$21.032328 |
| (c-1M) | A+B, 20 each × 1M | $41.878673 | $7.838660 | $5.653950 | $14.205675 | $6.241397 | **$75.818355** | −$15.818355 |
| (c-1.5M) | A+B, 20 each × 1.5M | $59.988506 | $7.924632 | $5.653950 | $14.205675 | $6.241397 | **$94.014161** | −$34.014161 |
| (d) | A, 20 × 1.5M | $32.623997 | $3.909216 | $2.826975 | $5.507291 | $6.241397 | **$51.108876** | **$8.891124** |

Totals use full precision. CPU reprojections preserve all frozen margins and
assay ceilings, plus the entire original $6.119213 fixed allowance. Each future
job now reserves ceil(body allocation)+361 seconds: two 120s startup/attempt
allowances, 60s termination, 1s polling, and six bounded 10s RPC allowances.
The original body startup margin is retained. Controls and diagnosis receive
the same repair; preparation is incurred and unchanged. No caching, faster GPU,
fewer erasure controls, reduced family panels, or halved global reserve is
assumed. See FORECAST_OPTIONS_REPAIRED.md and forecast_options_repaired.json.

Validation: **63 tests passed in 56.12s** with the prescribed interpreter,
bytecode and pytest cache disabled. CPU defect regressions were run against the
reviewed implementation before repair (seven failing semantic regressions;
lifecycle/horizon API and settlement regressions also failed before repair).
The GPU tests explicitly load the reviewed observer/classifier to reproduce
both Astra counterexamples alongside the repaired implementation. All GPU calls
were synchronous foreground bursts ≤60s with equal pauses; bursts are logged.
No push was performed. The user/coordinator must choose the cohort and approve
its changed estimands before any final registration/refreeze or launch.
