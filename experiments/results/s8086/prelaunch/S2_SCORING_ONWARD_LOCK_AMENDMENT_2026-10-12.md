# S2 operational amendment: scoring onward after ledger contention

The user's 2026-10-12 instruction authorizes finishing the stopped drafts,
validating them, refreezing, and relaunching scoring, extraction, the exact
eligibility commit and release, evolved L1/L2 assays, and constructed controls.
The 20 completed worlds are inputs. No worlds are rerun. World bytes, seeds,
horizon, scientific registration, scoring and eligibility rules, controls,
endpoints, resource rates, and non-world execution allocations are unchanged.

Each completed world records the prior operational freeze digest in its header.
The new freeze binds each of the 20 canonical input names, exact input SHA-256,
and that original digest. The scoring provenance gate accepts a prior digest
only for those exact completed bytes; changed bytes, a different prior digest,
or an unlisted world are rejected. The current freeze still verifies all
executing code. This operational lineage check changes no scoring calculation
or eligibility rule and requires no modification of a completed world header.

The previous scoring app stopped because nonblocking local ledger acquisition
raised BlockingIOError while the dispatcher held the ledger. That transient
accounting collision became terminal INCOMPLETE. Subsequent stop_required
checks labelled the terminal flag as a cap breach despite a complete projection
of $72.41250328898022, below the unchanged $95 cap. Its app meter was zero.

Each accounting poll now retries lock acquisition with exponential backoff
within a 250 ms allowance. Continued contention yields to observation and
retries on later polls, with a bounded 250 ms to 1 s interval. It never creates
a terminal row, uncertainty, or a stop reason by itself. Resource lifetimes
continue accruing; the current app's conservative contribution replaces its
stale contribution in the last complete projection, or retains the full plan
before the first successful replay. The parent supplies a verified complete starting
projection, including every carried plan extension, before creating a paid app;
first-poll contention therefore cannot omit historical spend. No new input receives an accounting permit
while writes are pending. A genuine cap or fixed app deadline still stops
independently of the ledger lock. Once the provider reports STOPPED and no tasks,
final accounting may wait for the lock without manufacturing a termination or
deadline failure. Confirmed stopping is released only after those rows persist.

SpendGuard reports the numerical cap flag, terminal dispatch flag, and unknown
physical liabilities separately. Lifecycle stop labels and host exceptions
report their actual cause. TaskList-only live observations preserve previously
accrued lifetime before advancing the observation time, including observations
that lack elapsed_s. They retain the same attempt identity. The real SIGSTOP
subprocess regression supplies PYTHONPATH explicitly and verifies that a
suspended lock owner cannot block the independent real-deadline stop.

Billing uses one poller lease per app and a workspace gate. A successful app
reading is preserved across rate limits, timeouts, unknown values and billing
lag. Polls start at 120 s and back off to at most 1,800 s. A nonblocking worker
keeps billing waits out of observation and stopping. Resource-time accounting
continues independently, and cumulative billing is overlapped once per shared
app. Billing failures do not supply zero-cost observations or cap decisions.

Before release, the entire suite and at least 1,000 concurrent randomized
sequences must pass. The production guard, watchdog accounting, and billing
poller are exercised together with ledger contention, rate limiting, recovery,
shared-app jobs, queues, preemptions, timeouts and failed control RPCs. Invariants
(a) successful completion within the complete budget never stops falsely,
(b) independent actual and conservatively charged spend remain within the cap,
(c) the whole cohort completes when its complete recovery budget fits, and
(d) cap stops are terminal INCOMPLETE with null scientific verdicts and forbid
further dispatch must have zero violations. Validation identities are bound in
operational_scoring_onward_validation_2026-10-12.json and the exact freeze.

The old hash-chained ledger and its terminal records remain unchanged. A new
ledger imports every old charged allocation and app attribution, preserving
failed allocations at their full bounds and finished worlds at their full
settled costs. The former scoring cohort and completed prebuild are historical
carry jobs; fresh scoring and an independently watched prebuild are separately
reserved. Historical app billing overlaps each app's retained cost once. The
single $5 reserve is retained once. The all-in forecast is $77.32835454898022
before any newly observed billing above those bounds, and must remain below $95
before release and during dispatch. No terminal flag is cleared in the old
ledger. The new release records the reconciled infrastructure cause explicitly.

Eligibility remains a separate commit of the exact 20 downloaded gzip files,
using explicit paths and the required Claude coauthor trailer, before its
current commit and per-world digests authorize assays. No push occurs. Only
operational code, tests, this amendment, validation evidence and the freeze are
included in the refreeze commit. Runtime receipts, ledgers and logs are retained
under the scoring-onward run directory and are not added to the commit.
