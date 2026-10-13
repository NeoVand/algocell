# Engine and continuity audit

Baseline: `6484a24`, `feat/app-legacy-base`. Read-only source audit; no GPU workloads were run for this audit.

## Why the current controls break continuity

`LegacyInstrument.svelte` has one long-lived `GPUEngine`, created in `onMount`. The processor selector calls `chooseMachine`, which calls `updateMachine(true)`. That helper always sets `playing = false`; its `true` branch calls `engine.configure`, which resets the population even when the tape and lattice dimensions are unchanged. Experiment selection follows the same route. The apparent engine replacement is usually a population reset; geometry changes additionally destroy and replace GPU buffers.

Other explicit pauses occur in `inspectCell`, checkpoint saving/restoring, and the callbacks supplied to the observation tools and encounter bench. `ObservationTools.svelte` calls its pause callback for confinement view, region selection and sampling, marker planting, history scrubbing/replay, and timelapse export. `EncounterBench.run` pauses before an isolated assay. The animation loop also suppresses simulation while any of `machineBusy`, `observationBusy`, `checkpointBusy`, or `cultureBusy` is set. Removing pause callbacks alone therefore does not restore continuous simulation.

The baseline also stops on document visibility loss and after a 55-second foreground run deadline. The latter is independent of UI controls and must be accounted for when interpreting long continuity tests.

## Existing engine capabilities

| Operation                                        | Current API                                                                   | Population, device, buffers and state                                                                                                             |
| ------------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mutation, step budget, pair budget               | `noiseExp`, `z80Steps`, `pairCount` setters                                   | Live uniforms on the next update; no reset                                                                                                        |
| Ablation                                         | `setAblation`                                                                 | Live uniform mask upload; no reset                                                                                                                |
| Core, compatibility, chemistry, reset convention | `setMachine` with identical geometry                                          | Same engine, GPUDevice, buffers, population, PRNG, update count and camera; recompiles pipelines and resets throughput estimate                   |
| Geometry                                         | `setMachine` with changed width, height, topology, tape length or ring length | Calls `changeGridConfig`; destroys/recreates buffers, initializes population, resets camera and update count                                      |
| New world                                        | `configure(settings, initial)`                                                | Validates initial bytes, calls `setMachine`, then `reset(seed)`; replaces population explicitly                                                   |
| Field view                                       | `setFieldMode`                                                                | Same world and buffers. Entering/leaving confinement recompiles pipelines to add/remove fetch instrumentation; other modes only update view state |
| Probe                                            | `readCell`                                                                    | Serialized one-tape readback, independent of the ongoing population update loop                                                                   |
| Region measurement                               | `readRegion`                                                                  | Samples the ordered GPU population into a staging buffer; computation can continue after the copy                                                 |
| Culture and assays                               | `runCulture`, `runAssays`                                                     | Separate Zilion machines and buffers sharing the existing GPUDevice; no population writes or replacement                                          |
| Camera, display, marker highlight                | `zoomAt`, `pan`, `setShowAverage`, `setBCS`, `setShowGrid`, `setMarker`       | View-only state/uniform changes                                                                                                                   |

All stored population elements are bytes. Z80, 6502 and 8086 accept the full byte domain as instruction memory; unsupported instructions have processor-defined runtime behavior. There is no additional byte-range conversion barrier for a same-length core swap. Core-specific ablation patterns and stack conventions must be normalized and validated before application.

Shader pipeline compilation is distinct from engine recreation. Confinement instrumentation and chemistry/core changes may rebuild pipelines on the same device and persistent buffers. Continuity assertions should track engine, GPUDevice and soup-buffer identity, world reset count and update count separately from pipeline revision.

## Required implementation boundaries

- Keep desired playback state independent of sidebar, tool, selection, readback and assay busy states. Only explicit Run actions change it. A short world transaction may withhold new submissions while preserving that desired state.
- Stage seed and geometry in the World form. Only its explicit new-world action may call `configure`. Same-geometry live settings call `setMachine`, with no `configure`, `reset`, or `clearMeasurements` side effect.
- Keep applied engine settings distinct from staged input. `setMachine` does not update the engine's live seed, which is correct for a live core/chemistry edit and unsuitable for a seed change.
- Serialize machine/field reconfiguration so quick successive control changes cannot race pipeline compilation or commit stale settings.
- Before destroying geometry buffers, stop new geometry-sensitive submissions and drain in-flight cell, population, hash and statistics reads. The baseline only waits for the GPU queue and cell reads.
- Make marker planting a GPU command. The baseline reads a 4-byte word to the CPU and later rewrites it; while running, this can overwrite three unrelated bytes changed in the meantime. A one-workgroup GPU byte patch, ordered between update submissions, avoids that race and avoids a population readback.
- Keep periodic probe reads bounded and discard stale target responses. A changed live tape should not remount its bench or erase already captured evidence; captured results retain their source tape and machine condition.

## Test instrumentation

Expose a read-only diagnostic snapshot on the existing development handle with engine ID, device initialization count, world reset count, buffer revision and pipeline revision. These counters need updates only at lifecycle boundaries, never in the simulation hot path. Retain object references in tests as a second identity check. Test every sidebar/view/tool route while the update count advances; a staged edit must leave applied settings and all lifecycle counters unchanged until the explicit action. In paused GPU tests, compare exact population bytes, PRNG continuation, update count and camera across live core swaps and chemistry edits.

## Existing benchmark harness

`scripts/evolution-benchmark.mjs` builds minified engine bundles and interleaves a frozen `BASELINE_REPO` with the candidate. It supports all five core contracts, compute/rendered modes, field modes, fixed updates, configurable warmup, reset drain, repeated alternating order, and idle pauses. It owns an ephemeral HTTP port and closes its browser/server in `finally`.

Use `6484a24` as the frozen baseline, every core contract, both compute and rendered modes, matched fixed workloads, `BENCH_ORDER=alternate`, `DRAIN_RESET=1`, and at least three repeats. Record raw runs and aggregate medians rather than treating a single noisy measurement as a regression.

`scripts/evolution-app-benchmark.mjs` additionally measures the actual app at 8 updates per animation frame, including normal UI statistics. It owns ports 4187 and 4188, closes pages between bursts, and cleans up its own servers. Its setup currently depends on the legacy Settings button and native Processor selector; adapt that setup for the unified shell while keeping timed work equivalent. `playwright.config.ts` owns port 4183 and runs one GPU worker. None uses the reviewer's port 5175.
