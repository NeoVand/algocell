> REGISTERED AND RELEASED: explicit user instruction; validation passed; $95 cap unchanged.

# S2 operational amendment — Modal v1 publication and parallel dispatch — 2026-10-12

The user authorized a volume filesystem repair, integrity-only archival and
step-0 recovery of world A/8086001, a minimal operational refreeze, and detached
parallel dispatch of the same 20 worlds under the unchanged $95 all-in cap.
This supersedes serial dispatch and the operational world timeout only. The frozen scientific registration,
world code, seeds 8086001–8086020, 3,000,000-step horizon, scoring/extraction,
L1/L2 rules, controls and scientific endpoints are unchanged.

Before reading any attempt files, the infrastructure decision was recorded in
`results/s8086/launch/run-20261012-volume-parallel/recovery_decision.json`.
The previous completed computation was never published because Modal v1
rejected `os.link`. It is discarded under the registered step-0 recovery rule.
Opaque size and SHA-256 metadata uniquely matched the failed `.publishing`
file to attempt `4a9e91370b8d48e5adfbd7ba8860b4c1`. Its original files remain
unchanged. Flat publication copies were renamed unchanged into that attempt's
`publication_failure_archive`, committed, reloaded and rehashed. No trajectory
records were parsed and no snapshot was decompressed. The new attempt uses
exactly seed 8086001 and begins at step 0.

A CPU-only function with a 60-second server body limit probed the actual
`algocell-atlas-runs` Volume before production changes. Same-directory rename
and replace, file fsync, directory open/fsync, local flock, chmod, parent mkdir,
exclusive create and rejection of an existing target, stat/exists/iteration,
binary/text/append/gzip I/O, unlink/rmdir and Volume commit/reload succeeded.
Hard links returned EPERM. Local flock and exclusive create are not distributed
locks; [Modal's volume documentation](https://modal.com/docs/guide/volumes)
explicitly excludes distributed locking and warns against multiple writers to
the same file. Each world has one submitted input and its own paths, with one
input per single-use container. Publication now reserves an exclusive marker,
checks canonical absence again, publishes snapshots and finally atomically
renames the fsynced complete record in the same directory. A stale marker fails
closed; a completed duplicate never overwrites the canonical history.

The exact repaired publication source was separately executed on the real
Volume with artificial complete records only. Complete record publication
survived commit/reload; a duplicate preserved bytes and a stale marker rejected
publication. Combined conservative preparation charges for the initial probe,
the integrity-selected archive probe, and repaired-publication verification
are $0.00118395, below the user's $0.10 probe budget. The first probe found
multiple historical attempt directories and made no archive mutation; selection
was corrected using opaque byte digests. All probe apps are confirmed STOPPED
with no tasks, and every call is charged in the new hash-chained ledger.

The dispatcher reserves all 20 world jobs before app setup, submits all 20
before any result wait, and uses one shared pool capped at 20 L4 containers.
Each world now has a measured 9,000-second (2.5-hour) server body limit. World 1
took about 63 minutes, measured as infrastructure timing before scientific
inspection; 9,000 seconds is about 2.38 times that measurement. World execution
code is unchanged. Startup remains 120 seconds, with the original two-startup
allowance, 120-second setup allowance and 121-second termination/poll/control
allowance. CPU/memory envelopes, single-use isolation and retries=0 are retained.
The dispatch reservation is 9,481 resource seconds, $2.39736566 per world.
Both live server liabilities and prospective reservations use the new limit. Modal queues inputs when
fewer than 20 GPUs are available. Scoring, extraction and evolved assays use
the same pattern across independent worlds. Constructed controls remain the
same single frozen call; the eligibility data commit/release still precedes
all evolved interventions.

One independent watchdog owns each active app and all its call IDs. App stop
and per-call cancellation are detached before bookkeeping/logging and take no
ledger lock. The cohort clock is fixed at startup, allowing serial queueing
within the sum of the amended lifecycle allocations plus the registered shared
world recovery envelope; preemptions and timeouts do not renew it. Every nominal
cohort reservation remains charged until provider STOPPED and an empty task list.
Every ended preemption adds its elapsed resource time and the original 121-second
termination/control allowance; unconfirmed vanished attempts retain a full
server liability for possible overlap. STOPPED plus an empty task list resolves
those vanished server liabilities, without releasing any cohort reservation.

The concurrent accounting defect was cleanup/body overlap: the app meter
includes cleanup after a container ends, but the old elapsed floor contained
body/startup time only. Adding the meter excess to a full retained cohort
reservation charged cleanup twice. The accounted elapsed floor now also contains
each ended attempt's conservative 121-second cleanup allowance, which already
sits in the cohort liability. Cumulative billing is overlapped against that
floor once per app; any billing beyond the accounted floor still increases the
forecast. Historical provider-confirmed stopped failed allocations use their
retained full cost as the accounted floor, keeping their reservations un-settled
and avoiding the same meter-plus-reservation duplication.

A world allocation timeout is prospectively classified as infrastructure failure
before any attempt-file inspection. Both the server FunctionTimeoutError and
the existing runner's exact allocation TimeoutError are handled. Up to the two
remaining registered recoveries may submit a fresh input with identical original
arguments, including seed, and step 0. Each recovery extends the complete plan
and reserves its full amended world envelope before spawn, under the same
watchdog and fixed clock. Partial/failed trajectories remain archived and unused;
the first complete durable trajectory rule is unchanged. Other execution,
publication, observation and scientific failures are terminal INCOMPLETE, with
null route/law verdicts. A cap stop forbids further dispatch.

All historical allocation bounds are carried forward: $4.499165268980222,
including the unchanged full $3.15139418 failed-world reservation, both prior
prebuild reservations and $0.00118395 for the real-volume probes. One of the
registered three original recoveries is consumed by that failed attempt; two
remain, now $4.79473132. The complete amended nominal allocation is
$68.71324183898022 before repartitioning historical preparation. With all
carried spend, the nominal complete all-in forecast is $72.41250328898022;
including both remaining recoveries it is $77.20723460898022 / $95. Restart
headroom is $22.58749671101978: nine full amended world envelopes fit
arithmetically, but only the two remaining registered recoveries are authorized.
There is $17.79276539101978 left after reserving both. Fresh historical billing
is $1.03704597 across ten provider-confirmed stopped apps; billing may lag.
The shared $5 storage/termination reserve is counted exactly once.

The measured complete spend forecast is $46.01391928898022. At 20-way
concurrency the measured planning ETA is 2.815 hours; using all conservative
body allocations gives 4.265 hours, including the retained diagnosis allowance.
Queueing and charged recoveries can extend these estimates. Non-world deadlines,
scoring/extraction/law budgets, prices and scientific schedules remain unchanged.
Exact recomputations and carry provenance are in the new
`results/s8086/launch/run-20261012-parallel-9000` directory.

Validation evidence is recorded in `operational_parallel_9000_validation_2026-10-12.json`
and its named exact artifacts. The full suite, original fault simulation and
new 20-world simulation must all pass before release. The latter uses the
production SpendGuard and Lifecycle with interleaved completions, preemptions,
restarts, slow startups, timeouts, failed stop/cancel RPCs, cumulative shared-app
billing and queues of 1/2/4/8/20 GPUs. The standalone 1,000 concurrent sequences
include the carried historical cost and produced zero invariant violations. Its
independent resource-lifetime oracle checks invariants (a)–(d), including
terminal cap stops and prohibition of further dispatch. The frozen scientific
files and original validation artifacts retain their exact prior identities.

Final validation: 976 passed, 0 failed, 3 unchanged historical skips; 702.02 seconds, exit 0. Original serial 1,000-sequence simulation and concurrent 1,000-sequence simulation both have 0 violations. The 750,000-row authenticated ledger check has 0 false stops. Real-volume publication proof matches the exact committed repair source.
