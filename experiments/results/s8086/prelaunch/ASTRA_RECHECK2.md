**Astra second S2 prelaunch recheck — 2026-10-12**

**Do not release this freeze.** The original suspended-ledger-lock defect and immediate far-JMP witness are repaired. The nominal forecast fits $85 under the coordinator's dynamic-enforcement standard. However, this review reproduces a residual false memory-provenance classification and two new operational failures in the dynamic guard: ordinary successful completion can trigger an irreversible cap stop, and normal ledger growth can exhaust the watchdog's 50ms accounting deadline. These findings do not depend on demanding a certified provider restart bound.

Reviewed HEAD `a07804534380cce62096e88ebcf6c61ecc6ab725`, both commits `17abd2f` and `a078045`, the entire diff since `e96abc0`, and `B5_DYNAMIC_REFREEZE_STOP_2026-10-12.md`. Paths below are relative to `experiments/`. This report is the only worktree file written. No commits, Modal/provider calls, GPU execution, population observations, paid prebuild or source repairs occurred. CPU tests used temporary scratch space, with bytecode and pytest caching disabled.

**Identity and validation.** All **270 manifest file hashes match**. Freeze SHA-256 is `353d57feec44548be1552704872aacf924b84249c9fe0aa99872e6ddee39c1c0`. The extracted option-(b) registration exactly matches the S2 portion of `REVISION_PREREG.md`. All 16 changed Python files compile without writing bytecode. Independent CPU runs cover **120 unique passing cases**: 23 B5, 68 B6 and 29 repair cases. The archived XML runs contain 930 passes/3 skips and 120 passes; after normalizing test-module prefixes their union is exactly **931 passes and 3 inherited skips**. The full GPU suite and 8,192 random-program GPU differential were inspected as archived evidence, not rerun. Plain `git diff --check e96abc0..HEAD` flags CSV CRLF endings; with `core.whitespace=cr-at-eol` it passes. This is formatting, not a launch blocker.

**B6: the narrow fix, instruction audit and differential are verified; complete provenance closure is not.**

Immediate far JMP now replaces CS provenance at `algocell_exp/census8086_trace.py:248`. The original parent `8e0e0e00ea0a000200900ef490900200` is correctly a literal writer with writing-episode starts `[0,36,42]`. The audit has **1,276 rows, all 256 opcodes and 38 classes**, with anchors checked against both vendored WGSL lengths. Per-class tests exercise representative ModRM encodings; this is useful coverage, not an exhaustive semantic proof of every instruction interaction.

Independent CPU replay of **all 21,182 archived encounters / 1,246 settings** verified trace hashes, current labels, ordered write channels and agreement with the separate flat value-tag oracle. There were zero differential mismatches. Comparison with `e96abc0` finds exactly **seven random-partner label changes**: three other-to-literal and four other-to-memory-fed. Every zero-partner label, all eight raw word tables, raw trace bytes, and non-classifier columns are unchanged. **07 A5 remains memory-fed in 68/68 encounters, with B fetches in all 64 random encounters.** The registered rationale counts and R0–R2 predictions remain unchanged. The archived 8,192 random programs report zero provenance and full-record/memory mismatches, with source identities verified by the manifest.

The oracle has its own flat tag bank and does not import the production classifier/spec. It nevertheless shares a source-selection assumption exposed by N1 below. Agreement between these implementations cannot close that common semantic error.

**N1 — BLOCKER: zero-count REP plus TF vector traffic is falsely classified as a memory copy.**

In [census8086_trace.py:202](/Users/neo/repos/algocell-wt-s2/experiments/algocell_exp/census8086_trace.py:202), MOVS takes the instruction step's first reads as its data source. Partner-write channels are appended at lines 209–216. The new six-store interrupt check at lines 263–269 invalidates only subsequent register state, after those channels have already been accepted. The oracle repeats this ordering at `i8086_value_tags.py:57,64–69,108`.

Concrete L16 parent and partner:

```text
A = eb06f49002f10000bc10009df3a5f400
B = 02010000000000000000000000000000
```

The parent jumps to A8, sets SP=16, and executes POPF. POPF reads B16=02/B17=01, normalizes flags to F102, and leaves SP=18. TF is sampled before each instruction, so POPF does not itself vector; the following `F3 A5` does. CX is reset to zero, so **REP MOVSW performs no string read or write**. Its post-instruction TF interrupt reads the vector words from A4–A7, then pushes flags to B16/B17, CS to A14/A15 and IP=14 to A12/A13. The vector destination F102 wraps to A2, which halts. These steps follow the unchanged WGSL at lines 195–209, 254–261, 477 and 522–528. Full starts are `[0,8,11,12,61698]`; the writing episode ends at start 12.

The IVT destination happens to equal the pushed flags. Both classifiers consequently return **memory-fed one-pass copier**, channels `['memory','memory']`. Those IVT reads did not supply the pushed flags. The high flag byte was transformed from the earlier POPF value 01 to F1, and flags/vector traffic is unsupported provenance under the conservative contract. The correct classification here is **other**, not an exact memory copy.

A CPU reconstruction executing this tape's small opcode subset from the vendored semantics confirmed the trace. The following shorter, no-write trace reproduction also demonstrates the defect:

```python
import numpy as np
from algocell_exp.census8086_trace import mechanism
from i8086_value_tags import value_channels, reference_mechanism

def inst(step, pc, bs, reads=(), writes=()):
    prefix = int(bs[0] == 0xf3)
    return ([(step, 0, (pc+i) % 32, v, 0) for i, v in enumerate(bs)]
        + [(step, 4, pc, bs[0], 0),
           (step, 3, pc+prefix+1, bs[prefix], 0)]
        + [(step, 1, p, v, 0) for p, v in reads]
        + [(step, 2, p, v, 0) for p, v in writes])

rows = inst(0, 0, [0xeb, 6]) + inst(1, 8, [0xbc, 16, 0])
rows += inst(2, 11, [0x9d], [(16, 2), (17, 1)])
rows += inst(3, 12, [0xf3, 0xa5],
    [(4, 2), (5, 241), (6, 0), (7, 0)],
    [(16, 2), (17, 241), (14, 0), (15, 0), (12, 14), (13, 0)])
log = np.asarray(rows, np.uint32)
print(mechanism(log, 16), mechanism(log, 16, audit_channels=True))
print(reference_mechanism(log, 16), value_channels(log, 16))
# Both: memory-fed one-pass copier; ['memory', 'memory'].
```

The `e96abc0` classifier also misclassifies this trace: **this is a newly discovered residual B4/B6 defect, not a claim that these commits introduced it**. It does not show that this particular tape passes culture or R2. It refutes the exact-source semantics used by the classifier. Separate the instruction's own bus transfers from post-instruction vector traffic, or conservatively reject the affected step before emitting its channels; add this interaction regression to both implementations and re-audit/refreeze.

**B5 against the coordinator's six criteria.**

| Criterion | Recheck finding |
|---|---|
| 1. Independent lock-free stop/cancel, including SIGSTOP lock holder | **Original defect closed.** Watchdog ledger access uses nonblocking flock and a 50ms whole-operation wait; stop and cancel launch independent processes before bookkeeping. Real SIGSTOP with the shared flock, blocked FIFO journal and blocked ledger-I/O regressions pass. See `lifecycle8086.py:86–112,232–239`. N3 concerns normal bookkeeping growth, not reappearance of the old blocking stop path. |
| 2. Accounted prebuild and setup inside the envelope | **Verified.** The watchdog starts before app.run; each app has 120s setup. A separate 600s timed prebuild reserves 1,081 total seconds and requires stopped confirmation. The receipt binds image ID, freeze and recipe; dispatch requires it in a fresh process and uses Image.from_id. See `prebuild8086.py:10–48`, `dispatch8086.py:150–156,196–199`. |
| 3. Server controls | **Verified in declarations and pinned SDK options.** Explicit execution/startup timeouts, retries=0, single-use containers, max1/min0/buffer0; dynamic invocations receive the frozen body deadline. See `modal_s8086.py:53–64`, `modal_score8086.py:20–22,73–75`, `dispatch8086.py:207,240`. |
| 4. Charge observations, retain liabilities/remaining work, stop on cap pressure | **Implemented and exercised, but operationally blocked by N2/N3.** Cursor generations retain short attempts; terminal attempts retain elapsed plus termination allowance; vanished unconfirmed attempts keep full liabilities. The ledger uses max(app metering, observed elapsed) plus remaining liabilities and all future reservations, avoiding double charging elapsed time. Cap/unknown/incomplete states prevent further dispatch. The reproduced defects fail closed; they are not demonstrations of an $85 overrun. |
| 5. Nominal single-attempt forecast fits | **Verified arithmetically:** $83.79381223898022, leaving $1.20618776101978. Recomputed projection, guard plan and deadlines agree. This forecast does not establish that the implementation can complete nominally in the presence of N2/N3. |
| 6. Provider-limit failure disclosed as residual risk | **Verified.** The amended registration explicitly discloses failure to honour server limits or app-stop/cancellation. No unconditional provider worst case is asserted or required by this review. |

The forecast is exactly the previous $81.03232817898021 plus **$2.4881424** for 82 existing app setups and **$0.27334166** for the separate prebuild. Original body allocations, incurred preparation and the full **$5 reserve** remain included.

**N2 — BLOCKER introduced in this diff: successful completion can create a fictitious restart reservation and stop the experiment.**

[Lifecycle.observe:139](/Users/neo/repos/algocell-wt-s2/experiments/lifecycle8086.py:139) reserves an entire next attempt whenever no attempt remains live and `work_complete` is false. This includes a successful completed container whose host finish acknowledgement has not yet been read. `watch()` snapshots the control file at line 254 before metering and task sampling; the dispatcher writes finish only after `call.get()` returns (`dispatch8086.py:161–170`). These channels have no ordering barrier. Even a finish written during the provider sample is missed until the next iteration. The adapter explicitly maps `TASK_STATE_COMPLETED` to `completed`, and single-use containers terminate after their input.

A CPU reproduction used the actual frozen plan, real SpendGuard/Lifecycle, one mock task, zero metering lag correction, and a 120s setup. For `world:A:8086001`, D=11,982s:

| Observation | Complete projection | State |
|---|---:|---|
| One live attempt | $83.76346903898022 | Allowed |
| Completed event with elapsed=D, empty tasks, finish=False | $86.82383361898023 | Stop/cancel; incomplete=True |
| Next poll: finish=True, app stopped and empty | $83.73312583898023 | Confirmed, but incomplete remains True |

The completion poll is at monotonic 12,102s, before the registered stop time 12,342s. At a successful elapsed time of only **5,000s**, the false projection is already **$85.05836509898022**. `wait_confirmed()` then raises at line 313, dispatch records a terminal incomplete state, and subsequent worlds/scoring cannot reserve. Correct durable output cannot override it.

The existing nominal-plan regression (`tests/test_b5_enforcement.py:230`) always supplies the completed event and finish=True together, missing this ordering. Reconcile successful input completion versus genuinely pending retry work, and test both event orders while preserving reservations for real unfinished inputs. Merely increasing the cap or allowing files to override a cap stop would not repair the state transition.

**N3 — BLOCKER introduced in this diff: normal ledger growth trips the 50ms accounting watchdog.**

[SpendGuard.transact:31](/Users/neo/repos/algocell-wt-s2/experiments/dispatch8086.py:31) rereads, parses and rehashes the entire shared experiment ledger for every transaction. Each poll appends liability and metered rows, while [Lifecycle.bookkeep:86](/Users/neo/repos/algocell-wt-s2/experiments/lifecycle8086.py:86) imposes a fixed 50ms deadline. Thus routine accounting cost grows with the complete run history.

Read-only CPU experiments constructed valid SHA-chained frozen-plan ledgers with production-shaped liability/meter events. Transactions took about 7ms after 1,000 polls, 22ms after 3,000, 45–50ms after 6,000 and 95–101ms after 12,000. A **16,003-row / 4,965,089-byte ledger representing 8,000 normal polls** reproduced the failure through actual Lifecycle.poll: the validated projection was **$83.76346903898022**, with stop_required=False; one live attempt had elapsed 8,000s, below its deadline. The poll issued stop and cancel after **0.06766s**, set incomplete=True, and journaled `TimeoutError: watchdog ledger IO exceeded 50ms`.

The precise crossing depends on host speed and RPC cadence. This ledger size is within the scale of one long world and accumulates across the twenty worlds and later jobs; it is not a malformed-ledger or provider-failure scenario. The frozen suite has no long-ledger regression. Use bounded/incremental accounting with authenticated history reconciliation, retaining independent stop and fail-closed handling; validate it at full experiment ledger scale before refreezing.

**Practical restart risk and cap recommendation.**

Recovery correctly restarts from step 0 with the same seed, isolates attempt directories, rejects partial histories and selects the first complete durable registered trajectory prospectively (`recovery8086.py:12–48`; `modal_s8086.py:80–94`). It cannot resume these checkpoints exactly because the full RNG/execution state is absent.

**The first preemption does not invariably stop the experiment immediately, but $1.21 is insufficient to absorb an arbitrary first world restart.** For a terminal-confirmed preemption, let e be the discarded attempt's elapsed resource seconds and s the app setup seconds. With D=11,982 and r=$0.00025286/s, the old terminal attempt plus pending replacement is charged:

```text
[s + (e + 121) + (D + 120 + 121)] * r
```

The original world reservation is `(D+481)*r = $3.15139418`. Extra exposure is therefore `(e+s-119)*r`. In an otherwise nominal ledger with s=120, the $1.206187761 headroom covers only **e <= 4,769.18s, approximately 79.49 minutes**. An earlier explicit preemption can fit financially. A preemption at 6,000s projects **$85.31122510** and stops the **whole experiment as INCOMPLETE**, preventing subsequent world/scoring reservations. If the old task disappears without terminal evidence, its full liability remains; observing the replacement projects **$86.85417682** under the same assumptions and stops immediately. These are intended conservative cap decisions, distinct from N2's false completion stop.

There is a second constraint: **extra cap alone does not provide restart time**. The app's fixed stop time is start+12,342s, with expiry at start+12,463s (`lifecycle8086.py:74–76`). Restart generations never renew this clock. A replacement needing the full 120s startup plus 11,982s execution can fit only if the first attempt consumed at most `240-s` seconds—**120–240s**, before additional gaps. Faster actual execution may leave more room, but late step-0 recovery cannot rely on the frozen full allocation. This is disclosed registered policy, not a separate violation of the coordinator's six criteria.

For planning, reserve one additional full world allocation per discarded world, preserving all present margins and the $5 reserve:

| Full-world-equivalent restarts | Extra allowance over nominal forecast | Required total cap | Increase over current $85 |
|---:|---:|---:|---:|
| 1 | $3.15139418 | $86.94520642 | $1.94520642 |
| 2 | $6.30278836 | $90.09660060 | $5.09660060 |
| 3 | $9.45418254 | $93.24799478 | $8.24799478 |

Rounded minima are **$87, $91 and $94**. I recommend **$95 total (+$10)** for three full-world-equivalent losses, leaving **$1.75200522** additional slack. This is a completion-risk recommendation, not a new release requirement or a promise against arbitrary restarts. To make it useful for late recovery, prospectively amend/refreeze a finite, charged recovery time budget alongside the cap; do not simply renew the existing app clock without reserving the added exposure. No cap or policy change is made by this report.

The A-only cohort, 3M horizon, schedules, source-root amendment, separate law-family estimands and decision thresholds remain unchanged in the reviewed diff. No additional blocker was found beyond N1–N3. Release requires fixing the residual provenance error and both nominal-operation guard failures, adding the corresponding interaction/ordering/scale regressions, and refreezing their evidence. A cap increase is optional under the stated acceptance standard and cannot substitute for those fixes.

RELEASE: NO-GO (residual B6 TF/vector false memory provenance; new B5 successful-completion cap-stop race and growing-ledger watchdog timeout)
