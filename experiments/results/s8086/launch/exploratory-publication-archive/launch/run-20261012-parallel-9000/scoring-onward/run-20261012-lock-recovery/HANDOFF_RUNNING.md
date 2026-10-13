# S2 scoring-onward relaunch — RUNNING

Refreeze commit: `4c0dbebe00f7f1a4fa2da8e8741ae52aa726e3bd`. No push. Commit includes only the 10 explicit paths
in `refreeze_commit_paths.json`, with the required Claude Opus 5.5 trailer.
Freeze SHA-256: `e98812d555b1632d07e5c561f786c1d129917f05eff676d87ea7acdbd722f3d9`; all 359 exact committed identities passed verification.

Validation: **992 passed, 0 failures, 3 unchanged historical skips**, full-suite
exit 0. All three billing fallback cases, the SIGSTOP/PYTHONPATH regression, and
the initial carried-projection contention case passed. The final concurrent
simulation has **1,000 sequences, 0 violations** of invariants (a)–(d),
8950 ledger-lock collisions, 7928 rate-limited billing reads, and
2330 rate-limit recoveries across 20 shared-app jobs. Final serial
1,000-sequence and 750,000-row ledger validations passed as part of the suite.
The final source fingerprint is `final_source_sha256.json`; source is unchanged
since the final full-suite start. Initial and second passing runs are also
preserved, but `full_tests.xml` and `full_tests.log` attest the final version.

The lock fix yields after bounded exponential retry and continues resource-time
accrual, using the complete starting projection including carried extensions.
No lock collision becomes terminal. Pending accounting withholds new dispatch
permits; a physically stopped app waits for final ledger rows without a false
termination/deadline failure. Numerical cap, unknown physical bounds, terminal
dispatch and deadline reasons have distinct truthful labels. One per-app poller
and workspace gate preserve successful cumulative billing through throttling,
timeouts and unknown reads, with resource-time accounting continuing independently.

World data were read only: 20 complete records and all 100 compressed snapshot
hashes verified. No worlds were rerun. All 65 other scientific source identities
are unchanged. The sole scoring-source diff is the operational freeze-provenance
gate: accept the prior digest only for the 20 exact byte-bound completed inputs.
Scoring calculations, eligibility rules, scientific schedules and endpoints
are unchanged. The new freeze binds each input SHA-256 and its original digest.

## Persistent processes and provider evidence

- Detached pipeline PID **92573**, its own session, reparented to PID 1.
- Scoring dispatcher PID **93348**.
- Independent scoring watchdog PID **94099**, its own session.
- Score app **`ap-oPclb4JJTIN2GSttSCXdz4`**, provider state `APP_STATE_EPHEMERAL`.
- Active variant **`fu-UwiLKUXDOBtkXZmBc1vL09`**, provider-confirmed timeout **437 s**,
  maximum **20 containers**. The named base function has different metadata;
  the concrete active variant is identified by AppGetLogs.function_id.
- At `2026-10-12T23:21:51.696806+00:00`, **20 inputs submitted, 4 running and 10 queued**.
  This reports actual available capacity, not a claim of 20 active containers.
- Watchdog is ready and current, permitting dispatch, with accounting_pending=false,
  incomplete=false, stop_required=false. Resource liabilities remain conservative.
- The accounted prebuild completed and stopped before scoring. Receipt:
  `prebuilt_image.json`. Image `im-mSaQN0cFDlOIVrYy14bD7A`; prebuild app `ap-HVdEYWxtneffRMm4w3ODZq`.

The detached launcher runs only score → extract → exact 20-file eligibility
Git commit and release → L1/L2 assays → constructed controls. Every future
commit uses explicit paths and the required coauthor trailer; no push. Eligibility
commit/release precedes all evolved assays. Keep this worktree's HEAD stable
while that commit/release executes; assay dispatch verifies the exact released
current commit. Stage PID files, events and watchdog prefixes change as stages
advance; the pipeline PID persists. No future stage depends on this conversation.

## Spend and ETA

Known cumulative metered spend: **$12.31932614**, billing may lag. The conservative
recorded-lifetime bound immediately before this relaunch was **$17.48499628**;
new app resource time continues to accrue independently in the watchdog/ledger.
All original allocations remain charged, including the old stopped scoring
cohort, prebuilds and all 20 fully settled world envelopes. The old ledger,
including its terminal rows, is unchanged and retained; no terminal flag was
cleared there. Historical app billing overlaps retained costs once. The shared
**$5 reserve is counted once**.

Complete all-in forecast: **$77.32835455 / $95**, headroom **$17.67164545**.
The fresh scoring cohort and separately watched prebuild are included. No world
recovery or world rerun is scheduled or needed. The source ledger copy and
`carry_forward.json` disclose every carried allocation and app attribution.

Conservative remaining ETA at the currently observed **4 slots**:
**4.45 hours**, around **2026-10-12T22:48:47.696806-05:00**. At eight slots, the same frozen
non-world envelopes imply **2.89 hours**; at twenty slots **1.33 hours**.
These estimates retain the conservative diagnosis allowance, exclude the
completed prebuild/worlds, and assume the observed capacity persists through
score, extraction and assays. Extra queue or recovery delays can extend them;
more allocated GPUs shorten them. Capacity is currently the timing limitation,
not the spend forecast.

## Paths and monitoring

Run directory: `results/s8086/launch/run-20261012-parallel-9000/scoring-onward/run-20261012-lock-recovery` (relative to experiments/).

- `pipeline.log`, `pipeline.status.json`, `pipeline.events.jsonl`, `pipeline.pid`
- `score.log`, `score.pid`, `spend.jsonl`, `carry_forward.json`
- `launch_receipt.json`, `prebuilt_image.json`, `launch_verification.json`
- `lifecycle-c9d22168ad82de2b.control.json` / `.status.json` / `.events.jsonl`
  / `.watchdog.log`
- `full_tests.log`, `full_tests.xml`, `test_runtime.json`
- `concurrent_fault_simulation.json`, `world_input_integrity.json`
- `provider_function_inventory.json`, `billing_snapshot.json`

Committed amendment and validation are under `results/s8086/prelaunch/`:
`S2_SCORING_ONWARD_LOCK_AMENDMENT_2026-10-12.md` and
`operational_scoring_onward_validation_2026-10-12.json`. Runtime files are
intentionally untracked. No forbidden path was committed.

```sh
cd /Users/neo/repos/algocell-wt-s2/experiments
R=results/s8086/launch/run-20261012-parallel-9000/scoring-onward/run-20261012-lock-recovery
ps -p 92573,93348,94099 -o pid,ppid,pgid,etime,state,command
cat "$R/pipeline.status.json" "$R/lifecycle-c9d22168ad82de2b.status.json"
tail -f "$R/pipeline.log" "$R/score.log"
```
