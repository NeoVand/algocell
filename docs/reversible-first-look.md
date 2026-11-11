# Reversible interaction rule: first look (2026-11-11)

Not an experiment. One run per setting, one seed, a short horizon, on a rule added the same day. It shows that the
rule runs and roughly where its behaviour changes. Nothing here is registered, replicated or reviewed.

## The rule

Each cell keeps two layers: its tape now and its tape before its last event. For a pair with tapes `X`, stored pasts
`P` and execution result `F(X)`:

    X' = F(X) xor X xor P        P' = X

This is invertible for any `F`. A cell is at rest when `P = X`, and then `X' = F(X)`, the ordinary write. Dissipation
is the chance, per cell and event, that the stored past is discarded (`P := X`) ahead of the event. At 100% the rule
is the standard overwrite; at 0% the world is closed and exactly reversible. Mutation is unchanged.

Code: `src/lib/gpu/shaders.ts` (write-back in `absorb_results`), `src/lib/gpu/engine.ts` (second layer, bound in the
spare slot 4), `src/lib/lab/types.ts` (`interaction`, `dissipation`). Under the standard rule the generated simulation
shader is byte-identical to `feat/app-unified` (SHA-256 compared for four configurations).

## What was run

Z80 compat 0.2, 148 x 135 square lattice, 16-byte tapes, ring 32, 128 steps, 8,192 pair slots, mutation 1 in 16,
seed 6, random start with every cell at rest. Driven headless through the dev hook (`window.__algocell.engine`):
`configure`, then `simulateStep` 2,000 times (8,000 for the last row), then `readSoupData`. Population replay is not
deterministic in this engine, so repeated runs will differ in detail.

| Rule | Dissipation | Updates | Distinct tapes / cells | Commonest tape | Its share | Zero bytes |
|---|---:|---:|---:|---|---:|---:|
| overwrite | – | 2,000 | 0.783 | `01 c5` repeated (the pusher) | 7.4% | 16.3% |
| reversible | 100% | 2,000 | 0.732 | `01 c5` repeated | 10.4% | 14.7% |
| reversible | 99% | 2,000 | 0.750 | `01 c5` repeated | 8.4% | 16.6% |
| reversible | 90% | 2,000 | 0.758 | `01 c5` repeated | 9.7% | 19.0% |
| reversible | 80% | 2,000 | 0.836 | `01 c5` repeated | 6.5% | 26.7% |
| reversible | 70% | 2,000 | 0.943 | `00 41` repeated | 0.9% | 37.7% |
| reversible | 60% | 2,000 | 0.951 | `00 41` repeated | 0.9% | 38.3% |
| reversible | 50% | 2,000 | 0.956 | `00 41` repeated | 0.8% | 38.1% |
| reversible | 30% | 2,000 | 0.957 | `00 41` repeated | 1.3% | 37.5% |
| reversible | 10% | 2,000 | 0.982 | `00 41` repeated | 0.6% | 25.1% |
| reversible | 0% | 2,000 | 1.000 | none repeated | 0.01% | 0.41% |
| reversible | 0% | 8,000 | 1.000 | none repeated | 0.01% | 0.37% |

For reference, the unmodified base branch gave 0.772 distinct and 16.5% zeros on the same settings.

## Reading, with the caution above

- At 100% the reversible rule behaves like the standard one, as the algebra says it must.
- Down to about 80% the familiar pusher still takes hold.
- Between 80% and 70% it stops appearing at this horizon. The soup instead fills with zeros (a quarter to nearly
  two fifths of all bytes) without any common replicator.
- At 0% nothing forms: the zero share stays at the random level (1 in 256 is 0.39%) for 8,000 updates.

## Known gaps

- The open/confined overlay compares the execution result with the tape, which is not the tape the reversible rule
  writes when a cell is not at rest. Its colours are unreliable under this rule.
- There is no view of the second layer and no count of cells at rest.
- Checkpoints do not store the second layer; a restored world starts at rest.
- A first version asked the device for an eleventh storage buffer and silently produced an invalid pipeline, so the
  lattice did not evolve at all. It was caught by comparing with the base branch, not by a test. There is no automated
  test of the new rule yet.
