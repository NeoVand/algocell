# Unified application performance

The baseline is `6484a24`. The engine candidate is `60cab9f`; the shell does not
replace the renderer or simulation kernels. Raw results include environment,
revision, settings, bundle hashes, every sample and the measurement method.
These are local Apple M4 / Chrome 155 measurements, not a claim about all GPUs.

## Engine harness

The existing `scripts/evolution-benchmark.mjs` runs baseline and candidate in
alternating order, three repetitions, 2,048 completed updates per sample,
64 warmup updates and a drained reset upload. Each sample uses a 200 × 200
square lattice, L16/P32, 8,192 pairs and 128 instruction slots. Rendered samples
draw every four updates. GPU bursts are short, with 2.5-second rests.

| Core           | Compute baseline → candidate, updates/s | Change | Rendered baseline → candidate, updates/s | Change |
| -------------- | --------------------------------------: | -----: | ---------------------------------------: | -----: |
| Z80 compat 0.2 |                           903.2 → 876.0 | −3.01% |                            868.9 → 881.0 | +1.39% |
| Z80 compat 0.3 |                           879.2 → 885.1 | +0.67% |                            838.1 → 846.3 | +0.98% |
| Z80 NMOS       |                           823.5 → 873.3 | +6.04% |                            809.5 → 821.2 | +1.44% |
| 6502           |                       1,849.5 → 1,838.6 | −0.59% |                        1,809.3 → 1,809.3 |  0.00% |
| 8086           |                           467.2 → 461.4 | −1.25% |                            431.2 → 473.7 | +9.84% |

Values are independent medians of three samples. The apparently large gains
are noisy and are not claimed as optimizations. The original Z80 compat 0.2
compute result missed the 2% gate and is retained in `engine-benchmark.json`.

## Investigating the Z80 result

Before another candidate comparison, an identical-baseline A/A control was run
with five alternating pairs, 512 warmup updates, the same 2,048-update timed
workload and three-second rests. Its medians were 873.3 and 876.9 updates/s
(+0.40%); individual pairs ranged from −1.15% to +2.20%. Both bundle hashes are
identical. See `engine-control-aa.json`.

The preselected five-pair candidate confirmation used that same warmed method.
Baseline and candidate medians were 864.2 and 867.9 updates/s (+0.42%); median
paired change was +0.65%. Individual pairs ranged from −8.29% to +3.04%, so the
evidence supports median parity rather than a universal sub-2% timing bound.
No shader or hot-loop optimization was made to obtain this result. See
`engine-z80-confirmation.json`.

`shader-contract.json` independently verifies byte-identical default generated
WGSL for all five cores. Lifecycle diagnostics and configuration serialization
do not add work to `simulateStep()` or the rendering loop.

## Full application

The final application comparison uses baseline `6484a24` and reviewed shell
`192a926`. `scripts/unified-app-benchmark.mjs` runs each application's actual
animation loop, renderer and controls at native 8× speed in a 1440 × 900 viewport.
Core and geometry are selected through the real UI and every applied setting is
asserted before measurement: 200 × 200, L16/P32, seed 6, 8,192 pairs, 128 steps,
mutation exponent 4 and benign zero. Each core uses matching native reset/dial
defaults, recorded per sample. Both applications are preloaded before timing.

There are three alternating repetitions per core, five-second bursts, one
completed warmup update, a final GPU drain and three-second rests. Every page
closes before the rest. All 30 samples are retained in `app-benchmark.json`;
there were no page errors. `app-benchmark-summary.json` contains the calculation.

| Core           | Baseline updates/s | Unified updates/s | Change |
| -------------- | -----------------: | ----------------: | -----: |
| Z80 compat 0.2 |              479.5 |             479.3 | −0.04% |
| Z80 compat 0.3 |              479.7 |             479.8 | +0.02% |
| Z80 NMOS       |              480.3 |             479.2 | −0.23% |
| 6502           |              480.7 |             479.8 | −0.19% |
| 8086           |              402.6 |             440.1 | +9.30% |

Values are independent medians of three complete samples. The first four cores
are predominantly limited by the native eight updates per animation frame at
about 60Hz; this is a default-use experience check, not maximum kernel throughput.
The 8086 samples vary substantially: baseline 330.7/402.6/405.5 and candidate
346.2/440.1/458.2 updates/s. The measured median gain is not claimed as an isolated
UI optimization or a general speedup.

Default UI policies and canvas areas remain authentic: the baseline keeps its
floating opcode chart visible, while the unified World sidebar leaves hidden
charts idle. Baseline canvas size is 1440 × 900; the unified canvas is 1024 × 808.
Neither application reads the full population during these bursts. The baseline
makes 288 small statistics readbacks across its fifteen samples; the candidate
makes zero while its charts are hidden. This does not imply zero readbacks when
the user explicitly probes, measures or opens Charts. The engine harness above
provides the separate comparison with matching GPU workloads.

## Gate and reproducibility

All five full-application medians meet the 2% regression gate; the worst change
is −0.23%. The existing engine harness's rendered medians all meet it, as do its
compute medians after the predeclared Z80 confirmation. The worst remaining
compute median is −1.25% for 8086. The original −3.01% Z80 result, the identical
baseline control and every confirmation sample remain visible above and in the
raw files. These measurements support local median parity, not a universal 2%
bound on individual timings.

```sh
BASELINE_REPO=/tmp/algocell-unified-6484a24 \
  BENCH_OUTPUT=docs/review/unified/engine-benchmark.json \
  MODES=compute,rendered FIXED_UPDATES=2048 REPEATS=3 \
  BENCH_ORDER=alternate PAUSE_MS=2500 WARMUP_UPDATES=64 DRAIN_RESET=1 \
  node scripts/evolution-benchmark.mjs

BASELINE_REPO=/tmp/algocell-unified-6484a24 \
  BENCH_OUTPUT=docs/review/unified/app-benchmark.json \
  node scripts/unified-app-benchmark.mjs
```

The baseline path must be a checkout of `6484a24` with dependencies available.
Run with an exclusive CPU/GPU window. The harness owns ports 4187/4188 and closes
its browsers and servers in `finally`; the reviewer server on 5175 is untouched.
