# S2 provider-limit preflight — STOP, 2026-10-12

**NO-GO. No replacement freeze or release.** The requested provider-side
worst-case liability cannot be certified below $85 from Modal's documented
limits. Stop under the task's explicit instruction: “if the worst case cannot
fit, stop and report.” This is a failed prerequisite, not a completed repair.

Astra's complete recheck was committed first, unchanged, in `4175f1d`.
Its SHA-256 is
`7d13613521eb721e2746732c1f79ccee065e541051c6ce1f1ee855c3f3040ffa`.

## Provider documentation checked

Official documentation, consulted 2026-10-12:

- [Failures and retries — Container crashes](https://modal.com/docs/guide/retries#container-crashes):
  infrastructure crashes during startup or execution reschedule containers and
  assigned work. Ephemeral apps stop crash retries when an unspecified failure
  rate is exceeded; no numerical maximum restart count per input is documented.
  Deployed apps retry crashes indefinitely with backoff. The application retry
  setting does not establish a finite infrastructure-restart limit.
- [Timeouts](https://modal.com/docs/guide/timeouts): execution timeout applies per
  attempt and startup has a separate timeout. The documentation also allows
  execution to exceed the nominal timeout by several seconds without stating a
  numerical upper limit. A per-attempt timeout does not cap aggregate attempts.
- [Preemption](https://modal.com/docs/guide/preemption): preempted functions restart
  on the same input. GPU functions cannot opt into non-preemptibility. This page
  supplies no maximum preemption restart count.
- [Retries reference](https://modal.com/docs/sdk/py/latest/Retries): an application
  retry policy is distinct from the infrastructure behavior documented above.
  The example using `max_retries=10` is not a documented maximum number of
  infrastructure restarts and must not be used as one.

**Maximum infrastructure restarts per input: no finite documented bound for
the existing ephemeral-app design.** This conclusion concerns what can be
certified from the documentation; it does not assert that Modal actually runs
every ephemeral input indefinitely. The explicitly indefinite deployed-app
policy must not be applied to the existing ephemeral app as if it were its
published policy.

## Liability calculation

CPU-only calculation from the unchanged `projection.json`, at its frozen rate
of $0.00025286 per resource second:

- 82 future jobs; sum of execution deadlines: 267,924 seconds.
- Startup limit: 120 seconds for each attempt of each job.
- Incurred preparation plus existing storage reserve: $5.799903818980222.
- Old client-watchdog forecast: $81.03232817898021. This is not a certified
  provider-side worst-case bound.

For an **illustrative assumed** maximum of `a` attempts per input, nominal
server-limit liability, before any new prebuild, setup, termination, idle time
or timeout overshoot, is

`5.799903818980222 + a * (267924 + 82 * 120) * 0.00025286`.

| Assumed attempts per input | Nominal liability |
|---|---:|
| 1 (no infrastructure restart) | $76.03530885898023 |
| 2 (one infrastructure restart) | $146.27071389898023 |

These assumptions are scenarios, not provider guarantees. One restart of every
input already exceeds $85 at the existing science-preserving deadlines, even
before omitted costs. With no documented finite attempt maximum, the requested
provider-only worst-case forecast has no certifiable finite value. Reducing
positive per-attempt reservations cannot supply the missing aggregate bound.
A functioning client watchdog would be useful but is not the requested
independent provider-side backstop when client controls fail.

Provider failure to honor its own enforced timeout would remain an explicit
residual disclosure in any future registration amendment. The absence of a
documented total restart bound is a separate unresolved prerequisite; it cannot
be recategorized as a provider failure to honor an existing total bound.

## Stopped work and verification

B5 remains open. No watchdog, setup/prebuild, concurrency, registration,
reservation or source changes were made after this prerequisite failed.
No image was built or published; the prohibition on paid Modal compute is
observed. No GPU execution, population launch, Modal API call or push occurred.

B6 remains open. No classifier repair, full write audit, independent provenance
differential, or census regeneration was completed. The witness is still an
outstanding blocker; no zero-mismatch claim is made. `07 A5` and the R0–R2
census rationale have not been reaudited in this task. R2 is unchanged.

The CPU preflight arithmetic and identity assertions passed. All 251 existing
freeze file identities still match, with zero mismatches. The full test suite
was not run because work stopped before implementation/refreeze; this report
does not claim an all-tests-passing replacement freeze.

Unchanged old manifest SHA-256:
`1b555fc267c94db3df3eb50e53d869f6cbf86c9eed6674ef920a2be316f0f340`.
Unchanged old projection SHA-256:
`ba4adf82a6327a2b73ebb6eb9c4aa1c9f4cab40fcde49a32c1588fb991d28182`.
The old files identify the rejected baseline only. Their historical
`launch_fits=true` field does not resolve B5 or authorize dispatch.

**STOP. Coordinator action requires a documented provider-enforced aggregate
restart/lifetime limit or an explicitly revised enforcement requirement before
repair/refreeze can proceed. No approval is requested by this report.**
