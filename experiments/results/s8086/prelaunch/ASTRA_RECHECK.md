# Astra recheck — refrozen S2 option (b)

2026-10-12. Reviewed `/Users/neo/repos/algocell-wt-s2` at `d0e684703139d370cc645eb1a6983d82d2a7e3c8`, including the changes since `df3d4b8`, repairs `15b6779` / `5309961`, STOPPED_REPAIR_REPORT.md, FREEZE_REPORT.md, the current S2 amendment, and corrected census evidence. Paths below are relative to `experiments/`; line references describe this HEAD.

**Do not release this freeze.** Option (b) fits the approved budget on paper, the L2 amendment is implemented, and the frozen census predictions survive the correction. However, the watchdog can block on the dispatcher's own ledger lock, defeating the physical spend bound. A new segment-provenance regression also admits a literal write as memory-fed. Passing the archived suite does not close either finding.

Only this file was written. No commits, Modal calls, GPU execution, population observations, or source/artifact repairs were performed. Parallel read-only audits covered lifecycle accounting, classifier/census evidence, and the L2 amendment. The recheck ran 25 focused existing CPU tests successfully with bytecode and pytest caching disabled, plus the read-only reconstructions and synthetic checks described below. The archived 840-pass suite was inspected, not rerun.

All **251** manifest file hashes match the worktree. The manifest itself is identical to its committed version, with SHA-256 `1b555fc267c94db3df3eb50e53d869f6cbf86c9eed6674ef920a2be316f0f340`. The extracted S2 registration copy exactly matches the S2 portion of REVISION_PREREG.md. These identity checks establish what was reviewed; they do not establish correctness.

| Finding | Disposition | Recheck conclusion |
|---|---|---|
| B1 — unfundable frozen design | **RESOLVED as a forecast defect** | User-approved A-only option (b), $85 cap, and complete $81.032328179 forecast agree. Physical enforcement remains blocked under B5. |
| B2 — incorrect L2 source origin/core/erasure prediction | **RESOLVED** | Registration, extraction, interventions, phase, serial ancestry, diagnosis and reporting implement the source-root amendment together. L1 controls retain their inputs and predictions. |
| B3 — inconsistent instruction-start addresses | **RESOLVED** | Observer records the hook's actual `pc`; ordinary and retained-REP starts now share the required convention. |
| B4 — unsound memory-copy provenance | **NOT CLOSED in full** | The original PUSH-SP, byte-pairing and implicit SP/SI/DI/CX defects are repaired. The replacement classifier introduces another false memory-origin label, documented as new blocker B6 below. |
| B5 — reservation not a physical lifecycle bound | **NOT RESOLVED** | The separate process shares blocking bookkeeping with the dispatcher; stopping can be prevented even with healthy provider controls. Setup and provider termination also lack the claimed complete bound. |
| B6 — stale CS provenance across immediate far JMP | **NEW BLOCKER** | The repaired classifier labels a literal PUSH-CS write as memory-fed after CS was overwritten by an immediate operand. |

## 1. Option (b), numerical scope and failure semantics

The amendment at `REVISION_PREREG.md:1759–1767` retains exactly 20 A worlds, L16/budget128, seeds 8086001–8086020, through 3M steps. It withdraws B/L32 population endpoints and comparisons prospectively. The active defaults, cohort, schemas, allocation plan, dispatcher and route/law report loops agree (`s8086.py:16–66`; `dispatch8086.py:162–206`; `score8086.py:192–219`; `law8086.py:259–308`). There are no B jobs or active B denominators/verdicts. Generic B machine/RNG identities and prospective parameter support remain for historical options and fixed L32 controls; the released-registration checks do not authorize a B population. Direct CPU calls to the default route and family verdict functions reject arm B.

A retains **585 samples**, checkpoints **0, 20k, 100k, 1M, 3M**, and exactly **11 R2b points from 2.75M through 3M at 25k spacing**. R0 is D≥10/20; R1 is O≥ceil(0.8D) when D≥10; R2a is ≥2/20; R2b is ≥1/20. The kill rule is still at least ceil(0.25D) confined first copiers in A with D≥10. The same-genotype persistence indicator remains separate, and scanning continues after both first events. These comparisons match the old registration and code; no threshold or horizon was quietly relaxed.

Summing the current liabilities and full $5 reserve gives **$81.03232817898021**, leaving **$3.96767182101979** below $85. All 82 future jobs reserve `(execution_deadline + 361) × $0.00025286/s`; incurred preparation is the 83rd liability. The guard plan equals the projection. Population, culture/route scoring, extraction, both law families, all ten controls, diagnosis and preparation remain included. This closes B1's numerical mismatch, subject to repairing B5 and recomputing any affected forecast.

The raw-record and statistical gates continue to prevent a truncated world from becoming a completed 3M negative. `score8086.py:96–136` rejects wrong horizons, absent/duplicate samples or checkpoints, corrupt snapshot identities, missing full summaries and any `incomplete` row. Unassessable qualifying cultures make the history unscoreable; 20 complete, scoreable registered worlds are required for route decisions. CPU checks confirmed that a missing world, an unscoreable world, or a short horizon makes every R0/R1/R2/kill decision null. Law reports retain incomplete/invalid and evolved-law-untested outcomes, unresolved counts, and independent-world coverage; L2 cannot rescue L1.

Runner initialization is now inside the error handler, which preserves the original exception without requiring another GPU read (`run8086.py:81–129`). Host terminal records provide a separate failure record (`dispatch8086.py:136–154`). Incoming metering now changes the same ledger transaction (`dispatch8086.py:91–95`), closing prior S1.

There is an operational qualification: the scorer/report APIs do not reconcile host terminal or watchdog records. A late failure after complete outputs have been persisted can coexist with those files. Complete files alone must not be used to declare an uncertain/cap-stopped job operationally complete; its terminal status needs reconciliation. I found no demonstrated path that scores a genuinely truncated record as a 3M negative, and do not count this qualification as a separate biological-censoring blocker. It does not cure B5 or make an unknown physical overrun acceptable.

## 2. B2: source-root amendment verified

The adopted text at `REVISION_PREREG.md:1771–1789` and `law8086.py:143–171` agree on raw K, the original-A root interval R, source core C_R, free roots F, erased coordinates E, mutable E\K and held E∩K. The code retains c≤d, checks the actual ordered map

`r(q) = s + ((L + q − s) mod d)`,

requires complete roots and preservation of the core at physical entry zero, and verifies map idempotence. Unsupported dependency transfers remain unresolved. Geometry must agree across all 17 baseline challenges; output agreement alone is not substituted for the dependency argument. Interventions and descendants use the actual source map, without rotation search or prefix repair (`law8086.py:187–197,310–325,394–426`). Diagnosis keeps raw dependencies separate from composed ancestry.

For L32/d16/s12 the independently rederived geometry is exactly K=0…8, R=12…27, C_R=16…24, F={12,13,14,15,25,26,27}, E\K={9,10,11,28,29,30,31}, nine held erased setup coordinates, source-seed phase 4 and physical entry zero.

The read-only CPU audit rederived **221 baseline traces across 13 amended certificates**. It reconstructed all **88,064** source12 intervention input/output hashes and serial partner bytes/hashes, checked every ancestry map and phase, and found zero mismatches. Assignment counts are 1,792 single, 512 joint and 1,792 erasure assignments, with eight actual offspring transfers. Comparing current generation with `df3d4b8` for all ten s=0 REP controls found **55,296 identical variant records**, including case identities, streams, mutated inputs and expected outputs. Free counts remain 2/10/26; the archived full control total remains 1,185,792 encounters.

These checks validate archived execution evidence and current CPU semantics, not new GPU execution. Arbitrary-loop proof coverage remains limited and explicitly unresolved; it is not silently converted into failed biology or supported law.

## 3. B3/B4 census invariance, including 07 A5

The start observer now uses `pc` (`algocell_exp/census8086_trace.py:44`), matching the ordinary and retained-REP hooks in `algocell_exp/machines/i8086_pair_maps_L16.wgsl:313,319–320`. The original REP-boundary, PUSH-SP, swapped-byte and implicit-register regressions pass on CPU.

Direct comparisons against `df3d4b8` establish that all eight exhaustive `words_*.csv` files are byte-identical. Across all four corrected mechanism tables, every row identity and every column except mechanism/evidence/trace hash is unchanged. **Every individual nonzero exact zero-partner word retains its mechanism label**, yielding the unchanged counts used in the registered rationale:

| L/budget | Nonzero exact period 1 | Nonzero exact period 2 | Literal | Memory-fed one-pass | Confined loop | Other |
|---|---:|---:|---:|---:|---:|---:|
| 16/128 | 0 | 20 | 12 | 2 | 0 | 6 |
| 16/512 | 0 | 29 | 12 | 1 | 0 | 16 |
| 32/128 | 0 | 26 | 11 | 5 | 0 | 10 |
| 32/512 | 0 | 26 | 11 | 4 | 0 | 11 |

The correction comprises exactly **1,476 label changes** among 21,182 encounters / 1,246 settings: 1,440 other→literal and 36 other→memory-fed one-pass. There are **122 changed zero-partner labels among non-exact candidates**. Therefore the correct invariance claim is about the frozen exact census counts and their contributing words, not every zero-partner label. The correction list discloses these changes. B9 51 changes eight random-partner labels, while its zero label is unchanged; FF 30 and AD 50 have no label changes. Both overflow rows remain other.

For **07 A5**, the audit loaded all **68 corrected archived traces**, matched each trace hash, reran the CPU classifier, and checked B-fetch events. All **17/17 encounters in each L/budget setting** remain memory-fed one-pass; all **16/16 random encounters** fetch B and copy exactly, with full B coverage, zero measured original-partner accesses and zero final-output difference. It remains the registered open memory-fed witness. This result and the census invariance do not prove classifier correctness for arbitrary evolved tapes.

## 4. New BLOCKER B6: an immediate overwrite leaves stale CS provenance

`algocell_exp/census8086_trace.py:225` now tracks a memory load into CS. But line 240 treats immediate far JMP (`EA`) as neutral, although the machine explicitly overwrites CS with its immediate operand (`i8086_pair_maps_L16.wgsl:512`). A subsequent PUSH CS uses the stale segment source at classifier line 199, and line 209 accepts coincidental byte equality.

Concrete L16 parent, zero partner, normal reset SP=1:

`8e0e0e00ea0a000200900ef490900200`

1. `MOV CS,[000E]` reads A14=02 and A15=00, setting CS=2.
2. At full PC 36, ring position 4, `EA 0A 00 02 00` sets IP=10 and **replaces CS with immediate 2**.
3. At full PC 42, ring position 10, `PUSH CS` stores 02 at pair position 31 in B and 00 at position 0 in A.

The full starts are distinct: `[0,36,42]`. The B write is literal, sourced from the immediate far-JMP operand. Current code returns **memory-fed one-pass copier**. The `df3d4b8` classifier returns **other** on the same trace, establishing a new false-positive regression introduced during repair.

The following no-write CPU reproduction uses events consistent with the vendored MOV-CS, far-JMP and push semantics (`i8086_pair_maps_L16.wgsl:415,512,150,452`):

```python
import numpy as np
from algocell_exp.census8086_trace import mechanism

def inst(step, pc, bs, reads=(), writes=()):
    return (
        [(step, 0, pc % 32, bs[0], 0),
         (step, 4, pc, bs[0], 0), (step, 3, pc + 1, bs[0], 0)]
        + [(step, 0, (pc + i) % 32, v, 0)
           for i, v in enumerate(bs[1:], 1)]
        + [(step, 1, p, v, 0) for p, v in reads]
        + [(step, 2, p, v, 0) for p, v in writes]
    )

rows = inst(0, 0, [0x8e, 0x0e, 14, 0], [(14, 2), (15, 0)])
rows += inst(1, 36, [0xea, 10, 0, 2, 0])
rows += inst(2, 42, [0x0e], writes=[(31, 2), (0, 0)])
print(mechanism(np.asarray(rows, np.uint32), 16))
# ('memory-fed one-pass copier', 'paired_memory_moves_distinct_starts')
```

This does not claim that the example passes culture or itself satisfies R2. It refutes the promised write-provenance semantics used by R2. Correct the CS overwrite, validate the regression and other admitted segment transfers, re-audit affected classifications, and refreeze. The original B4 witnesses passing cannot release this replacement classifier.

## 5. B5: the watchdog still does not enforce the complete $85 bound

The repair adds real safeguards: explicit 120-second startup timeouts, zero idle/buffer containers, per-function maximum one container, a monotonic clock shared across attempts, a separate watchdog process, cancellation/app-stop requests, and settlement only after STOPPED plus no tasks. Unknown bounds prevent settlement/new reservations once recorded. The six-RPC margin covers the examined deadline-crossing polling sequence under the assumed RPC and termination limits; that sequence alone is not an overrun finding.

**Concrete failure with healthy provider controls:**

1. `bounded_call()` spawns the remote input at `dispatch8086.py:130`, then calls `guard.start()` at line 132.
2. `SpendGuard.transact()` acquires blocking `flock(LOCK_EX)` at lines 29–30 and keeps it through flush/fsync at line 106. A suspended dispatcher, or one hung while holding this lock, remains a live PID.
3. The watchdog's immediate/periodic billing update calls the same blocking transaction at `lifecycle8086.py:162`, **before** reaching `state.poll()` at line 166.
4. It can now wait indefinitely for the dispatcher-held lock. It cannot reach the deadline/attempt checks, cancel/stop requests, or overrun reporting. The live-PID probe does not detect suspension.

Thus the process is separate, but its stop path is not independent of a blocked dispatcher. Per-attempt startup and body timeouts cannot bound total restart lifetime: repeated container imports despite `retries=0` are already recorded at `results/s8086/prelaunch/benchmark_failed_attempt.json:18–19`. Finite reservations and $3.97 headroom do not bound that continuing physical liability. The independence test at `tests/test_s8086_repairs.py:181–210` exercises Lifecycle without the shared SpendGuard lock and misses this failure.

Two additional coverage limits reinforce the same B5 finding:

- The watchdog is created only after entering `app.run()` (`dispatch8086.py:126–128`), and its clock begins at `lifecycle8086.py:183`. Installed Modal 1.6.1 resolves/publishes app members and can build/wait for images before that context yields. The configured image includes build operations (`modal_s8086.py:13–19`). This setup is outside the watchdog clock. This review did not measure cold-build billing, so it does not assert a particular setup overcharge; it finds no complete setup bound.
- Bounded RPC waits limit the local request duration, not provider resource lifetime. `lifecycle8086.py:96–106,152–170` keeps requesting stop and recording unknown overruns if termination cannot be confirmed. Neither the frozen code nor evidence establishes a provider-side total-lifetime cutoff or an enforced maximum termination delay. The amendment expressly acknowledges that assumption at `REVISION_PREREG.md:1805`. Honest unknown accounting is useful, but is not physical enforcement of a hard cap.

Before release, make the cancellation/stop path runnable independently of dispatcher-held ledger locks and blocking bookkeeping, and test a suspended parent holding the actual shared lock. Cover all resource-incurring setup, attempts and termination with an enforceable complete envelope, including the disposition when control requests fail. Reforecast and refreeze any changed allocation. Recording an unconfirmed overrun after it occurs does not satisfy B5.

The positive amendment, census and incomplete-input checks above stand. They do not outweigh unresolved physical spend enforcement or a newly unsound R2 classifier.

RELEASE: NO-GO (B5 physical spend bound remains unenforced; new B6 stale-CS false memory-fed classification leaves B4 unclosed)
