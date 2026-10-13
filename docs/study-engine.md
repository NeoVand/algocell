# Zilion migration study

Read-only source study, 2026-10-13. This supports `APP_PLAN.md`; it is not a claim that browser measurements or acceptance tests have already run.

## Sources read

Zilion 0.3.1: README, `docs/ARCHITECTURE.md`, `docs/I8086.md`, `examples/soup.html`, `src/world/spec.ts`, engine/runner/result/chunk types, manifest, root exports, Z80/6502/8086 descriptors, and `test/world/algocell.mjs`. App: `CLAUDE.md`, package metadata, all `src/lib/gpu/`, all `src/lib/sim/`, `z80-disasm.ts`, `z80-opcodes.ts`, and checkpoint implementation. Source trees outside the app worktree were read only.

## Findings that affect the implementation

1. Zilion separates core, world and population host. The package does not own pairing, mutation, genealogy or population measurements. The app should keep those definitions explicit rather than suggest a machine manifest identifies a whole experiment.
2. `createEngine` → `engine.machine` → `Machine.run` or `Machine.batch` is the supported new API. `Z80_CORE_WGSL` and the old facade remain byte-identical 0.2 exports, so the existing optimized view can remain available during the migration.
3. Z80 **must** select compat. `0.2` means frozen historical semantics; `0.3` fixes R refresh counting, WZ/MEMPTR, IM, and block-repeat flags. `zilog-nmos` is only valid with 0.3 and adds Q. Label all three choices explicitly. The engine package version and core semantic version are different provenance fields.
4. The NMOS 6502 and coherent-fetch 8086 are exported on npm. BFF exists in the checkout/exporter but is excluded from npm files and root exports; do not offer a runnable BFF choice.
5. On 8086 explicitly pin `params: { prefetch: 'coherent', hidden_latch: 'zero_ea' }`. No prefetch queue or timing claim is supported. REP is one step per iteration; its saved decode state matters when resuming. The paper stack phase is SP=1 for P=32/64, produced by `firstPushAt: 'lastByte'`; SP=0 is a distinct experiment.
6. `topAliases: 'lastByte'` is the paper Z80 convention, not generally the same as the old web app's hardcoded SP=65535. They agree for P=32, but old hex P=38 maps SP to 23 while topAliases maps it to 37. A legacy-web world must pin `init.set.sp=65535` if it promises old hex encounter behavior.
7. Historical encounter equivalence is stronger than historical population replay. The old web shader's atomic claims race by scheduling and mutation writes perform non-atomic read/modify/write of shared words. A seed alone cannot establish exact full-soup replay. Preserve frozen core behavior and expose the new deterministic host scheduler/version; do not call it a reproduction of a registered cohort.
8. The current CPU trace is an approximation that disagrees with the actual GPU: `sim/z80.ts` handles DD/FD as HL and treats block I/O as no-op. It has no R refresh increment in `step`, no modern WZ/Q, and no suppression or lethal conditions. Do not use it to classify execution under the new engine. Inspect an encounter through the same Zilion machine that runs the soup.
9. Existing Z80 disassembly is incomplete: DD/FD are shown as `DD ..`/`FD ..` and many ED instructions are raw bytes; missing operands are often invented as zero. New disassemblers should expose truncation instead of implying data exists beyond a selected tape.
10. Existing `family:writes-mem` is not strictly all memory-writing Z80 instructions under coarse ablation: indexed DDCB/FDCB operations write memory even for register destinations, while the coarse CB page marks only z=6 as a writer. An optional all-writers intervention must block these decoded forms as well (fine pages or conservative all corresponding CB rotate/SET/RES entries). This requires a probe, not merely counting selected opcodes. The paper's preset named “no-copy” is different: exactly all LD families plus stack, exchange and block-copy; CALL/RST and read-modify-write operations remain available. Preserve that registered set under its historical name.

## Concrete API contract

```ts
const engine = await createEngine();
const machine = await engine.machine({
	core: { isa: z80, compat: '0.2', variant: 'classic' },
	world: {
		memory: {
			kind: 'segments',
			segments: [
				{ bytes: L, name: 'A' },
				{ bytes: L, name: 'B' }
			],
			ring: P
		},
		init: { stack: { topAliases: 'lastByte' } },
		budget: { unit: 'steps', value: 128 },
		ablation: { table: 'uniform' },
		outputs: { memory: true, state: true, fetchMap: true, startMap: true, writeCounts: true }
	}
});
const result = await machine.run(lanes, { seed, budget: 128, ablation });
result.memoryOf(i);
result.stateOf(i);
result.writesOf(i);
result.fetchedOf(i);
result.startsOf(i);
result.haltReasonOf(i);
result.trapCauseOf(i);
```

The host `Uint8Array` is tightly packed A then B (`2L` bytes/lane). GPU segments are independently padded to u32 words; let Zilion pack them. P may exceed 2L, in which case padding starts at zero each new encounter and is never written back to the population. Address arithmetic wraps at ISA width before the world ring mapping; a non-power-of-two ring has a meaningful address seam. `'lastByte'` means the end of the last loaded segment, not the end of padding.

Initialization applies reset, stack convention, entry, constant set, random fields, then raw onReset. Required: explicit stack convention/set/per-lane state and explicit budget. Defaults are recorded but avoid relying on them for published presets. 6502 default P=0x20 and 8086 FLAGS=0xf002 are reset state, not immutable flags.

### Host service shape

Use a new `src/lib/lab/types.ts` for serializable settings and a plain TypeScript `engine.ts` service (GPU devices do not belong in deep reactive proxies). A Svelte rune state facade can own status, controls, selected cell and chart history. The service owns tight population bytes, deterministic scheduler RNG state, generation/encounter counts, reusable resident lane capacity, exact tape counts, last initiating-encounter observations and manifests. Configure/reset must serialize against in-flight updates; stale compile or culture results must not replace a newer selection.

First host scheduler: seeded serial pairing with four reflected square neighbors and non-overlapping claims, separate seeded mutation draws and separate culture RNG. Seed zero must remain valid. Mutation is random replacement, so some mutations leave the byte unchanged. Count accepted encounters and actual returned executed steps; do not label attempted pairs × budget as executed instructions.

### Resident batches, chunking and lifecycle

`const batch = machine.batch(capacity, { maxLanesPerSlab })`; `batch.upload(lanes)`; `batch.run({ lanes, seed, budget, ablation })` gives results while reusing allocations. `batch.encode(encoder, options)` permits chaining future GPU population kernels through exposed `batch.slabs[*].buffers`; `batch.read({lanes})` snapshots submitted work. Each encode must be submitted before another encode with different options because uniforms are written at encode time. A CPU gather/scatter first pass still transfers lane data each update: residency saves allocations, not those transfers.

`Machine.run` automatically chunks by workgroup limits, per-binding limits and staging-buffer limits; global lane indices keep random initialization/dials invariant across chunks. `maxLanesPerChunk` is useful for regression testing. Batch has equivalent slab partitioning. Zilion README reports example CPU-host soup around 150 updates/s on an M4, with transfer overhead dominant; this is upstream evidence, not an app benchmark. Benchmark the actual instrument in bursts under 60 seconds and report accepted encounters/s.

Destroy batch before machine, then engine. A machine queues its `run` calls and lets already-called work complete on destroy; batch.destroy releases buffers immediately while submitted work completes. Add device-loss/error handling. Rebuild only when core/world compile-time settings change; budget and ablation masks are per-run uniforms. Limit cache size or destroy old machines.

Resumption uses `machine.run(previous.memory, {resume: previous.record, stepBase, budget})` or `batch.run({resume:true,stepBase,budget})`. It requires steps budget plus state and memory outputs and no private state in raw hooks. Preserve padding, maps and counters through the record. This is encounter continuation, distinct from advancing a soup update.

### Chemistry and exact historical dials

`halting.lethal: { firstUnit: 0, probability: p, salt: 0 }` judges the raw first byte, before prefixes and ablation. Zilion's built-in dial uses a counter hash and `round(p*2^32)`, invariant to chunking and resume. The historical paper Z80 dial used PCG and floor thresholds. For exact paper behavior use the raw-hook mapping already validated in Zilion `test/world/algocell.mjs`: private state seeded with `zil_seed*2654435761 + zil_lane*2246822519 + 0x165667b1`, one warmup, draw only on zero first units, threshold floor. Its stateful hook is not resumable.

Historical random-register initialization likewise uses a registered PCG hook, seeded `zil_seed*2246822519 + zil_lane*3266489917 + 0x27d4eb2f`, warmup, then A,F,B,C,D,E,H,L,A2,F2,B2,C2,D2,E2,H2,L2 low eight bits and IX,IY low sixteen bits. Default `init.random` is a modern independent counter hash and must not be passed off as that exact arm.

### Per-ISA ablation and disassembly

Uniform words are page-major, bit key&31 in word page\*wordsPerPage+(key>>5). Z80 default coarse is three 256-bit pages (24 u32), compatible with old masks; fine is seven pages. 6502 and 8086 have one 256-bit opcode page (8 u32). 8086 uses the final opcode after segment/LOCK/REP prefixes. A skip consumes only the opcode (6502) or prefixes+opcode (8086), so operands may execute next; the UI must say this rather than imply the whole original instruction becomes an equal-length NOP.

Reuse app Z80 opcode family data with the strict indexed-writer correction. 6502 `M6502_OPCODES` includes mnemonic and addressing mode for all 256 entries; `m6502.table()` gives lengths/families but no operands. Implement a local table-based formatter. 8086 descriptor supplies no mnemonic table or disassembler; a local decoder must handle prefixes, ModRM displacement, immediate lengths and opcode groups. Unknown or unsupported forms must be clearly raw/unknown and consume only known bytes. No new runtime dependency is necessary.

### Measurements

Fetch maps include opcodes, prefixes, immediates and displacements, not data reads. Start maps include instruction starts. An initiating A encounter is confined when every fetched position is below L, open otherwise (including padding); this is encounter-specific and not proof of copying or heredity. Store the condition/partner and sample age, classify only A, mark modified/absorbed partners unknown, and invalidate after mutation. Do not paint B as if execution started in B. A whole-lattice assay against a fixed partner is a different measurement and needs a separate label.

Culture selected tape against a fixed, separately seeded panel of fresh random partners, with mutation off and the current machine conditions. Report original-phase exact partner copies, best cyclic identity and the declared ≥75% heuristic separately; expose partner/output bytes, fetch map, writes, state and halt/trap. A serial transfer assay feeds a selected offspring into another fresh partner; marker carriage must be scored separately from lineage survival. Never call a single successful encounter or tape abundance a heredity result.

Charts: zero-byte fraction has denominator N\*L (exclude padding); top-tape share uses exact byte sequence identity and N denominator; diversity should be defined explicitly as distinct exact tapes/N (with absolute count available), optionally Shannon diversity later. Hash colors are visual identifiers only; a 32-bit hash collision must not merge exact diversity counts. Histories need bounded storage and clear reset markers.

### Checkpoints and exports

Current PNG v1 includes dimensions, seed, noiseExp, budget, pairCount, suppression patterns and batchCount but does not store RNG state, machine identity or world; app restore cannot promise exact continuation. Preserve v1 parsing for the legacy view and label its semantics. Add a new version with settings, scheduler id/version/RNG state, tightly packed bytes and layout, generation/counts, mutation algorithm, runtime masks, manifest SHA-256 identities, selected tape and assay data where appropriate. Validate lengths/bounds/ISA/version before replacing live state. A frozen snapshot import is distinct from replay.

Export the actual `machine.manifest` (`zilion.manifest/2`, core/world/WGSL/options SHA-256), settings and source preset provenance. The manifest world hash alone does not hash runtime budget overrides, masks, seed, lane order, population scheduler or mutation state; include all of those in an experiment export. Existing v1 loader scans for IEND backwards and trusts gzip lengths/JSON; add size, shape and range validation before extending it.

## Verification recommendations

No-GPU tests: world stack/ring/ISA selections, preset exact parameters, ablation mask page bounds and strict writer coverage, disassembler known examples/truncation, deterministic scheduler and restored RNG, exact metrics and checkpoint validation. Browser GPU tests in short bounded bursts: Z80 pusher copy, displacement edit, selected culture, one 6502 and one coherent 8086 run, chunk invariance, frozen 0.2 known result, fetch confinement, every preset starts and increments encounters, device-unavailable UX. Keep existing differential dev harness available as a legacy diagnostic, with an explicit compat label.
