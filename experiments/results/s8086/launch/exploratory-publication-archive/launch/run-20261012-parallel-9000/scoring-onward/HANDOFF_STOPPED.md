S2 scoring-onward handoff: STOPPED on explicit test-failure condition.

No refreeze, commit, push, prebuild, world dispatch, or scoring-onward launch occurred.
All completed world data and scientific/scoring/eligibility/endpoint rules are unchanged.
Unvalidated changes remain in the working tree; see stopped_diagnosis.json for exact files and pending work.

Exact original failure: lifecycle8086.py HEAD lines 104/109 propagate a transient LOCK_NB
BlockingIOError. Score watchdog events lines 21/22 record observation/accounting unavailable
and Errno 35. The dispatcher then writes terminal at ledger line 5524. SpendGuard HEAD line
202 includes experiment_stopped in stop_required; lifecycle HEAD lines 208/214 label any
stop_required as exceeding cap. The numerical projection never exceeded $95: it was
$72.41250328898022. Last score meter was $0. There were no meter_error events in that
watchdog and its stderr log is empty. Shared-app costs were already grouped once.

Billing recovered on one batched query: billing_attempt_1.json. Combining each app once
with previously verified historical app costs gives $12.28942546 known cumulative billed
spend (billing may lag). A conservative recorded-lifetime bound including historical full
allocations is $17.48499627932854. Keeping all original charged allocations plus a fresh
score cohort and necessary prebuild gives an all-in forecast $77.32835454898022 / $95.

Targeted validation: 39 passed, 4 failed. Three new fallback cases expose zero resource
accrual for raw TaskList-only samples: observe() synthesizes elapsed_s=0 each poll and
resets observed_at. One existing SIGSTOP test failed because its subprocess lacked
PYTHONPATH and could not import dispatch8086. No 1,000-sequence validation was run.
The production concurrent simulation is extended with actual AppMeterPoller throttling,
unknown reads, timeouts and persistent no-meter sequences, but remains unreleased.

The drafted fix adds one app poller lease/worker, a workspace RPC gate and exponential
backoff; failed reads preserve the last valid meter and rely on independently accounted
resource time. It also retries brief ledger lock contention within 250ms and distinguishes
numerical cap, unknown physical liability and terminal dispatch reasons. This draft must
be repaired and validated before release. Stop condition prevents further work this turn.

No new PIDs or ETA exist. Prior pipeline/world/watchdog PIDs 70164/70931/71708 are gone.
One non-destructive audit check was appended to the original hash-chained ledger; its
terminal state has not been cleared. No worlds were rerun or modified.
