# Final S2 preparation and replacement freeze — 2026-10-12

**STOPPED before launch. No Modal call, paid compute, population trajectory,
launch receipt or push occurred in this preparation. Coordinator final
independent review is pending.**

The prospective R2 withdrawal and approved $95 cap were committed as
`39de676` before the full census re-audit and before population data. R2a/R2b
are descriptive and their verdict fields are null. Every detected copier has
all 16 causal encounter labels/checks, four class counts and every other case
with a reason. R0, fetch-based R1, the kill rule, L1 and amended L2 remain
confirmatory and unchanged; A remains 20×3M and B withdrawn.

The complete census re-audit covers 1,246 settings / 21,182 encounters.
Exactly 1,247 labels change: 1,175 literal writer→other and 72 memory-fed
one-pass→other. All 18,601 other cases and all changes/counts are published in
`../../census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md` and its CSVs. Raw word
tables, original full records, memory and archived traces are unchanged.
The 8,192 random-program scalar prefix/expression/class differential has zero
mismatches. Longest census burst: 0.739119s; longest random
burst: 0.290238s. All bursts are below 60 seconds.

The N2 completion race is repaired in both event orders. Incremental ledger
replay authenticates the prior prefix, verifies new rows and retains rollback
isolation; a bounded 10-second accounting allowance replaces 50 ms. Real
locks, blocked I/O and independent stop/cancel paths remain tested. The
750,000-row / 216,643,537-byte full-cap scale regression passes
with zero false stops: 3.171025s cold and
0.341510s incremental reconciliation.

Forecast: **$83.793812238980 / $95**, all-in including preparation
and $5 storage/termination reserve. Headroom: $11.206187761020.
One full-world envelope: $3.15139418 / 12,463s. Three full-world restarts cost
$9.45418254; forecast plus three is
$93.247994778980, leaving
$1.752005221020. Each world has a fixed
49,852s lifetime including its recovery allowance; restarts do not renew it.
Recoveries share the all-in cap and use the unchanged same-seed step-0 and
first-complete durable selection policy. No provider restart-count bound is
claimed.

The cloud timings remain historical legacy-classifier measurements. Causal
verification is included within the unchanged bounded scoring allocation;
no new cloud causal-throughput measurement or completion guarantee is claimed.
Exhausting any allocation or stopping at the cap yields INCOMPLETE and null
scientific verdicts, never a negative. No cases or law denominators are dropped
to make the budget fit. Provider failure to honour server/stop limits remains
the explicitly registered residual risk.

The production-state-machine simulation ran **1,000 randomized
sequences with 0 violations** of invariants (a)–(d): no successful-completion
false stop within forecast; charged and actual spend within cap; all worlds
complete when the conservative budget allows; cap stops are INCOMPLETE.
830 sequences complete and
170 stop as INCOMPLETE. Both asynchronous event
orders, startup/termination delays and preemptions are covered. Actual resource
seconds are accumulated separately. Real I/O and ledger scale are checked
separately from the simulation's in-memory I/O adapter.

Full suite: **959 passed, 3 inherited skips, 0 failures**, in
701.73s; the expanded ledger case additionally passes after
its row-count increase. Complete JUnit reports, burst journals and validation
identities are archived. Longest full-suite GPU burst:
3.532221s. The three inherited skips concern
vendored-machine regeneration against a different Zilion checkout.

`freeze.json` is the authoritative replacement manifest, attested by the same
Git commit. It binds this report, the registration/schedules, classifier,
corrected census, differential, simulation, complete tests, cost plan, machine
exports, dependencies and exact file identities. Its own hash is excluded to
avoid a cycle and is reported to the coordinator after commitment. Previous
review receipts do not release this freeze. **STOP; do not launch.**
