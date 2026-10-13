# S2 operational amendment — noninteractive stopping — 2026-10-12

The user authorized this operational repair and a fresh detached relaunch of
the complete previously released pipeline. The prior launch stopped during
prebuild; no population, scoring, extraction or retention call was dispatched.

The only production source change is adding `--yes` to the frozen
`lifecycle8086.py --stop` invocation of pinned Modal 1.6.1's `app stop` command.
Its watchdog deliberately disconnects stdin, stdout and stderr. Modal's
confirmation prompt aborted in that environment. `--yes` skips that prompt;
the independent stopping, cancellation, observation, deadlines, settlement
requirements and retained-liability discipline stay unchanged.

`tests/test_modal_noninteractive8086.py` exercises the direct stop entrypoint
and the provider's detached stop subprocess chain against a fake executable.
The executable checks all three descriptors are non-TTY, rejects the missing
confirmation flag, and records the accepted command. A separate zero-cost
test passes `--yes --help` to the real pinned CLI and parses the advertised
confirmation option. A frozen-source inventory assertion covers every Modal
CLI invocation. The audit of SDK calls is archived with this amendment.

No science, population inputs, seeds, horizon, classifier, eligibility,
interventions, L1/L2 denominators, controls, endpoint or spend rule changes.
The original complete design allocation, resource rates, future job plan,
three-full-world charged provider recovery allowance and single $5 shared
storage/termination reserve remain frozen. The hard cap is $95 all in.

The fresh operational directory is `results/s8086/launch/run-20261012-relaunch`.
Its new hash-chained ledger imports all five historical apps' cumulative
metered spend, verified STOPPED states and empty task lists. The historical
preparation allocation is represented by its four original app settlements;
the aggregate preparation placeholder carries zero additional liability to
avoid charging that allocation twice. The prior prebuild's original
$0.27334166 reservation remains un-settled under a separate historical job.
The original terminal ledger is preserved byte for byte and is never resumed.
All imports use existing SpendGuard event types; no accounting code changes.

The retained historical preparation bound is $0.7999038189802217. Including
the original failed prebuild retains $1.0732454789802217, before the single
$5 shared overhead. Historical metered spend is $0.13979978, with billing lag
disclosed. The complete nominal relaunch projection is $84.06715389898022;
adding the unchanged $9.45418254 recovery allowance gives $93.52133643898023.
The cap and remaining-work gate govern all future dispatches.

The replacement manifest binds this one-line repair, its tests and fresh
validation evidence. Earlier scientific artifacts and their original identities
remain historical evidence. The renewed launch receipt identifies the new
freeze and the explicit user authorization for this operational amendment.
The prior independent design GO remains the scientific review; no new
independent scientific review is claimed.

Relaunch uses the accounted prebuild, then the frozen serial dispatcher for
all 20 worlds, scoring and extraction. A detached orchestration invocation
downloads the exact eligibility gzip bytes, commits only those explicit data
paths with the required coauthor trailer, and writes the hash-bound eligibility
release before the frozen L1/L2 assays and constructed controls. No push or
host retry is performed. Every paid job has the existing independent watchdog.
Any failure stops the orchestration and retains the frozen failure liabilities.
