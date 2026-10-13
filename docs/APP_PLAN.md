# Algocell app plan: Zilion 0.3.1

Written 2026-10-13 before the first implementation pass. Workspace: `algocell-wt-app`; branch `feat/app-zilion-0.3`, starting at `4c84fec`. No push or edits to the paper checkout, other worktrees, or main are authorized. This document records the intended product and the first implementation boundary; verification and final deviations will be appended after implementation.

## Study record and audit

The team read every file under `src/`, including the 6,199-line main page, GPU orchestration and generated shaders, local CPU interpreter, opcode model and disassembler, persistence helpers, developer differential harness, scaffold test, README and CLAUDE.md. Detailed evidence is in [app study](study-app.md). Zilion's README, architecture and 8086 documents, soup example, world specification, engine/result/manifest types, and core descriptors were read from the local 0.3.1 source; see [engine study](study-engine.md). The manuscript, story talk, registrations, machine definitions, stages, cohorts, provenance and figure contract were studied read-only; see [paper study](study-paper.md), which supplies the exact catalogue and source references below.

`npm ci` succeeds. The baseline runs at `http://127.0.0.1:5174/` (5173 was occupied). A short local GPU burst produced a visibly changing lattice and approximately 500M reported instruction slots/s. This is a local observation, not a benchmark of executed instructions. Pause, Settings, Help and About worked. Baseline screenshots are in `docs/screenshots/baseline-*.png`. Baseline `npm run check` had zero errors and 20 accessibility warnings. Baseline lint failed on existing formatting (`src/app.html`, main page, layout CSS), plus the generated Playwright snapshot directory. The only existing end-to-end test checks the scaffold demo heading.

### Features worth keeping

- Immediate, dense, spatially faithful soup; square and hex topology; byte detail and average-cell views; pan, zoom and fit.
- Run/pause/reset, speed, seed, lattice dimensions, mutation, pair count and budget controls.
- Prefix-aware Z80 ablation, custom saved presets, top-byte and exact-tape composition, diversity.
- Hover/tap inspection, species exemplars, colormaps, brightness/contrast/saturation, grid toggle.
- PNG-plus-binary `.acell` snapshots, drag-and-drop import, keyboard shortcuts, and an independent developer differential harness.

### Problems with evidence and consequence

- One component mixes lifecycle, settings, GPU operations, metrics, disassembly, help, export and nearly 2,000 CSS lines. It is already in runes mode; the issue is ownership and coupling, not merely syntax.
- The first screen has no visible name, question, legend or explanation. Small icon buttons and a floating chart overlap the very data they control. Help has seven tabs and asserts one simplified emergence story.
- The paper is about the separation of reproduction and transmitted variation. About currently describes generic Z80 artificial life; Guide calls mutation “bit flips” although the algorithm replaces bytes and asserts a particular copier without qualification.
- Full soup readback on pointer movement, hidden-tab-dependent chart collection, and whole-tape hashing without provenance make interaction expensive and measurements presentation dependent. Hex byte normalization counts padded storage rather than actual bytes.
- Viewport-dependent lattice dimensions change a run's world. Seed alone is not an identity. GPU atomic claiming and packed-word mutation races also prevent promises of cross-device bitwise replay for old soups.
- The old checkpoint omits complete random-generator and scheduler continuation state and has weak structural validation. Its bytes remain valuable, but importing them is a snapshot continuation, not exact historical replay.
- The local CPU interpreter is not a validated oracle for modern assays. The Z80 disassembler has incomplete indexed/ED coverage. New assays must run the selected Zilion machine, not that interpreter.
- Existing chart marks, dialogs and form labels trigger accessibility warnings. Small targets, hidden overflow and hover-only detail are weak on mobile and keyboard.
- The old main page loads a KaTeX stylesheet from a CDN. The new app will make no external runtime calls. Static diagrams do not need Mermaid/KaTeX execution.

## Product vision

Algocell is both a public window and a small, inspectable research instrument. Its defining experience is to move from a visible pattern to a falsifiable question about one tape, without confusing pattern, reproduction, confinement and transmission.

| Audience                                   | In five minutes                                                                                          | In an afternoon                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Curious public                             | Run a soup, select one cell, see why a repeated pattern is not yet evidence of life or heredity          | Compare benign and lethal chemistry; inspect a pusher and a confined copier                           |
| Students                                   | Change one rule, read a labeled chart, perform a fresh-partner self-copy test                            | Keep paired seeds/configurations, export observations and explain the limits of an assay              |
| Origins-of-life / evolution researchers    | Identify the processor, reset, ring, mutation and budget; distinguish observations from functional tests | Culture candidates, manipulate displacement and markers, compare controls with full manifests         |
| Astrobiology / artificial-life researchers | Switch instruction chemistry while seeing the exact world convention                                     | Explore stack phase, ablation and geometry; export a complete condition for larger registered cohorts |

The app is not a substitute for registered multi-world experiments. A browser session is an illustrative world. A negative short assay is not a proof that a tape never reproduces.

## Information architecture and interaction design

The primary view is a research desk, with the spatial lattice as the visual anchor. Desktop has a calm header with Algocell, Guide, About and the legacy instrument; an experiment/control rail; a dominant canvas; a tape inspector; and aligned measurements below the canvas. At intermediate widths the inspector moves below; at mobile width everything follows a single reading order with a useful canvas before long details. The page scrolls normally.

```
Algocell                         Guide  About  Legacy
Experiment / world       live lattice             selected tape
Preset + explanation     run / step / reset        bytes / disassembly
ISA + chemistry          byte / fetch layers       fresh-partner culture
Seed + advanced rules    spatial legend            assay interpretation
                         measured histories
```

Start paused with a seeded random soup and an invitation to run. No invisible GPU task starts from reading documentation. Run is bounded to 60 seconds per burst, pauses in hidden tabs, and can be restarted. Step performs one complete update. Status distinguishes compiling, ready, running, paused, failed and disposed; no overlapping reset/readback/culture operations. Structural settings apply transactionally and restart the world, rather than silently rewriting a live run. Ablation is a per-run uniform table and can be updated without recompiling the core; UI changes still need explicit semantics and an event record.

Preset cards/selection show: what changes, source, evidence status, what to look for, exact paper parameters and any app departure. A scientific preset is not a promise that emergence occurs during a short demo. Seeded mechanism presets clearly say “constructed demonstration”. Changing ISA selects a valid initial stack convention and clears incompatible ablation selections.

The inspector is usable by clicking/tapping or selecting a cell by number with a keyboard. It shows exact bytes, starting offsets, per-ISA static disassembly, and the selected tape's current context. Static disassembly is a linear interpretation, not an execution trace. Fresh-partner testing reports exact copies, best cyclic similarity, open/confined fetch status, write counts and halt/trap reasons when available. A marker demonstration is separately labeled as value carriage; value equality does not establish ancestry.

Measurements sample independently of panel visibility. Zero fraction uses actual bytes. Top-tape share uses exact full byte strings, not hash equality. Diversity is labeled with its mathematical definition (unique tapes / total tapes, and optionally Shannon entropy/effective number). Charts show update x-axes, 0–100% reference levels, units, text endpoints and explanatory labels. “Species” becomes “exact tapes”; identity is not a functional class.

Exports: versioned JSON checkpoint with core/world configuration, initial-condition definition, soup bytes, update/encounter/executed-step counts, host RNG state, scheduler version and manifest SHA-256 identities. CSV measurements and manifest exports require no server. Checkpoints do not yet retain the preset/source label, chart history, fetch observations or assay results; a combined research archive is a next-phase addition. Input validation precedes allocation or configuration changes. Legacy `.acell` remains accessible in the legacy instrument with an honest compatibility note. Imported settings always pause. File operations are local.

## Design system

Use the paper's figure contract as the semantic foundation: Arial throughout; white canvas and surfaces; charcoal `#25313A`; pale grey `#E6E8EB`; open/unclassified context `#83909A`; transmission/full-tape T teal `#168A8D`; erased/partial R vermilion `#D85A3A`; lethal-only violet `#8653A8`. ISA is written as text, never encoded using heredity colors. Confined execution gets a charcoal boundary, not transmission teal. Unknown cells stay visibly unknown. Color legends always include words or shapes.

The byte lattice retains physical lattice order and sharp nearest-neighbor bytes, with zero near-white and stable colors for nonzero values, following `figures/soup_stills.py`. Genotype coloring is explicitly an identity map, not an R/T map. No fabricated archived trajectory is presented as paper evidence.

Typography: body 14–16px with 1.5 line height; useful controls at least 14px; secondary notes at least 12px; restrained 24–32px title. Scientific values use tabular numerals. Spacing uses 4, 8, 12, 16, 24, 32px. Panels exist to group tasks, not to make identical cards from every paragraph. Borders and shadows are quiet; the lattice supplies the visual complexity. Contrast must remain readable on white; colored text is not the sole signal.

Controls have explicit labels, visible focus, disabled/busy states and at least 40–44px comfortable targets where practical. Semantic buttons/selects/inputs, dialog focus handling and Escape close, linked headings, reduced-motion support, and keyboard alternatives are required. No decorative animation; motion only from an explicitly running world or a direct interaction. Errors explain the next action without erasing the current snapshot. No web fonts, telemetry or network assets.

Design review: a generic dark simulation HUD would repeat the current discovery problem. A generic white dashboard of statistic cards would diminish the organism-like spatial subject. This plan instead uses a large faithful lattice with a small amount of persistent scientific context and a single coherent inspector.

## Zilion 0.3.1 migration and module boundaries

Dependency change: upgrade existing `@neovand/zilion` to `^0.3.1`. No new runtime dependency is proposed. Existing Playwright handles testing; native Canvas and SVG handle rendering/charts; own small ISA disassemblers use Zilion's opcode descriptors where available. Existing Mermaid/KaTeX may remain installed for the archived view until removal is independently justified, but no CDN resource is allowed.

- `lib/lab/types.ts`: serializable world/app contract; ISA, compat, geometry, chemistry, seed, budget and observation types.
- `lib/lab/engine.ts`: one engine/device, compiled machine, reusable resident batch, host scheduler, gather/execute/scatter/mutation, deterministic host PRNG state, measurements, culture and cleanup. It contains no Svelte or DOM state.
- `lib/lab/session.svelte.ts`: per-page reactive class; lifecycle, run cancellation, immutable snapshots, status and actions. Large typed arrays use raw state and explicit versioning rather than proxies. No global cross-user mutable singleton.
- `lib/lab/presets.ts` and content: source-bound experiments and readable scientific interpretation.
- `lib/lab/disassemble.ts` (or per-ISA modules): safe bounded linear decoding for Z80, NMOS 6502 and 8086; truthful unknown/truncated bytes, indexed and variable-length forms handled without pretending static code is a trace.
- Small Svelte components: ExperimentControls, SoupViewport, TapeInspector, Measurements, documentation views. Main page composes them.
- Legacy page/component retains the old optimized GPU path, appearance controls, stored presets, legacy checkpoint and developer harness. Label it and do not run it concurrently with the new desk. Exact old-core semantics remain available there and in new Z80 compat 0.2 machines.

Use `createEngine`, `engine.machine({ core, world })`. Z80 compat must be explicit (`0.2` frozen historical, `0.3` corrected); `zilog-nmos` remains selectable as an independent variant. NMOS 6502 and 8086 `nmos` use their advertised semantics. 8086 uses `{ prefetch: 'coherent', hidden_latch: 'zero_ea' }`; this is not a silicon prefetch-queue timing model. BFF is in the source tree but is excluded from npm exports, so no fake BFF menu item or copied core.

Worlds use two named L-byte segments on a P-byte ring, zero initialized padding, explicit reset/register rules, step budgets, optional lethal first-unit byte and probability, `ablation: {table:'uniform'}`, and outputs for memory/state/fetchMap/startMap/writeCounts. `topAliases` and `firstPushAt` are different interventions: historical Z80 top aliases the last byte, 6502 first push is at the last byte, 8086 first push gives SP=1 at P32 and a word across end/0. Do not transpose a register value between ISAs and call it the same geometry. The old hex app uses literal SP=65535, which differs from last-byte alias at P38.

Ablation UI follows each core's decode layout and instruction table: Z80 base/CB/ED pages, 6502 opcodes and 8086 whole opcodes. The first implementation groups an 8086 opcode's ModRM suboperations together; selecting individual group operations is deferred. Empty, no-block-copy and strict-no-copy presets have distinct documented meanings. Per-run uniform tables avoid compilation for each checkbox. Explicit opcode selection is preferable to ambiguous mnemonic substring matching.

Resident batches reuse GPU allocations. Large lane work follows advertised chunk limits; chunked machine runs preserve lane indices and deterministic dial draws. Initial desk uses deterministic host pairing and tight host bytes with GPU execution: simpler inspectable correctness first, at a throughput cost compared with the historical all-GPU lattice. Describe this new scheduler in exports. It does not recreate archived GPU race ordering. Preserve core compatibility exactly; never promise bitwise old population replay. Next phase can move gathering, claiming, mutation and measurements onto the GPU after equivalence tests.

Fetch overlay records the last measured _initiating_ tape encounter, with unknown for unmeasured or invalidated cells. Fetching any position beyond A (partner or padding) is open. An outcome is not a permanent property of a genotype: partner, reset and budget matter. Start maps distinguish instruction starts from operands. Culture runs many fresh partners using an independent seed, without altering soup RNG, and separates exact copying from similarity and continued reproduction. Full registered branching/serial protocols remain a separately specified next phase; the small assay must not claim their result.

Marker planting: support a selected site/value in isolated culture and an explicit forward-transfer demonstration; display parent, offspring and original control. Track equal value only unless ordered-write ancestry is actually implemented. Spatial soup marker tracking requires ancestry tags and ordered write tracing, not recoloring equal bytes; plan it, do not mislabel a value heatmap.

## Paper experiments catalogue

The complete source-level catalogue is in [study-paper.md](study-paper.md) and will be kept beside executable preset metadata. Its parameters are authoritative; the following specifies coverage and user-facing framing.

Common historical world: reflecting 160×125 lattice (20,000 tapes), 8,192 drawn neighbor pair slots/update, 128-step encounters, k=4 ⇒ 512 replacement attempts/update, reset registers zero, historical Z80 compat 0.2. Drawn slots differ from collision-free active pairs; replacements may hit the same position or replace a byte with itself. App host scheduling is explicitly distinct. Exact registrations include seeds, horizons, cohort arms and analyses that a single short browser run does not reproduce.

| Preset                        | Contract and source                                                                         | Explanation and what to inspect                                                                                                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Historical stage G L16        | Stage G run definitions; L16/P32, historical Z80, common world                              | Order and repeated byte patterns can precede detected reproduction. Compare zero fraction with a fresh-partner culture test; do not treat visual takeover as a detection criterion.                                       |
| Benign / lethal zero          | Registered substrate intervention; identical baseline except `firstUnit:0`, p=0 or 1        | Zero accumulation changes when zero is fatal on instruction start. A zero used as data/operand is not automatically lethal. Observe trajectories across multiple seeds, not a promised winner.                            |
| Lethality dial                | Registered probability grid, source-listed seeds/horizon                                    | Hold everything else fixed while varying halt probability. A dial changes chemical dynamics; it is not a continuous “difficulty” setting.                                                                                 |
| Random registers              | Registered randomized reset fields, stack and entry preserved as specified                  | Tests dependence on zero-reset bias. Register noise is distinct from mutation and must be seeded per encounter.                                                                                                           |
| Ring geometry L12             | L12/P24 versus L12/P28 padding, exact stack phase from registered definitions               | Same tape length with a changed ring can alter overwrites and closure. Padding is zeroed per encounter and does not return to tapes.                                                                                      |
| No block copy                 | Four Z80 ED keys A0/A8/B0/B8                                                                | Removes dedicated copy instructions while retaining stack and ordinary loads. It is not absence of every mechanism able to copy.                                                                                          |
| Strict no-copy                | Exact load/stack/exchange/block-copy family mask from source                                | Removes the named instruction families; other memory writers (CALL/RST, read-modify-write) remain. “No-copy” names an ablation, not a proof that copying is impossible.                                                   |
| 6502 S1 arm A                 | Registered L16/P32 NMOS 6502, firstPushAt lastByte, budget128                               | A byte-push processor changes the available machinery. Read the arm-specific outcome and limitations from the source catalogue.                                                                                           |
| 8086 S2 arm A                 | L16/P32, nmos/coherent/zero_ea, SP1, flags61442, budget128; 20 seeds8086001–20/horizon3M    | S2 detected 14/20 worlds: two open and 12 confined; the open-first prediction failed. Neither registered evolved law assay had eligible candidates. Do not include withdrawn L32 as an S2 result.                         |
| Offset-switch R/T competition | Frozen seeded designs and offset switch from registration                                   | Designed competition asks about ecology given specified founders; it is not spontaneous discovery. Display initial R/T bytes, proportion and changed phase.                                                               |
| Displacement edit             | Z80 core `04 5e ed b0` versus a full-tape distance byte (e.g. `50` at L16/P32) with payload | Both can be fetch-confined while the first overwrites payload and the second transmits it under the stated forward-copy conditions. Read output bytes and marker carriage rather than deducing heredity from confinement. |
| Pusher / return closer        | Frozen example tapes (separate examples, no invented ancestry)                              | Run each against fresh partners. An open pusher and confined return architecture illustrate mechanisms; their side-by-side display is not an evolutionary lineage.                                                        |

## About rewrite: outline and full draft

Outline: question → encounter model → paper's bounded results → how the instrument tests claims → Zilion and semantics → authorship and antecedents → reproducibility and links.

**Copying is not yet inheriting.** Algocell is a living view of a machine-code soup. Each cell holds a short sequence of bytes. Neighboring tapes meet in shared memory, a processor runs their instructions, and the resulting bytes return to the lattice. Random byte replacements keep introducing variation. There is no assigned fitness score. Programs spread by writing into the tapes they encounter.

A regular pattern is an observation, not a diagnosis. It may arise from zero-writing instructions, a short copying mechanism, or other dynamics. To ask whether a tape reproduces, we take it out of the soup and let it meet fresh partners. To ask whether a change is inherited, we introduce that change and follow it through reproduction. These are different experiments.

The paper studies how reproductive survival, confinement of instruction fetching, and transmission of variation can come apart. Some tapes reproduce while repeatedly reconstructing a small pattern, erasing introduced differences. Other mechanisms transmit a larger payload. In a forward-copy family, an inherited displacement controls how much variation can survive, subject to explicit geometric and execution conditions. Cross-processor tests delimit that mechanism's scope; they do not establish a universal law for all digital or chemical life.

The controls here expose the world's assumptions: processor, historical or corrected semantics, ring, reset state, time budget, lethal-byte chemistry, mutation and ablation. The fetch layer reports where instruction-stream bytes were read. It cannot show all sources of data dependence or establish heredity. The small culture assay is an inspection tool, not the paper's full registered branching and serial-transfer protocol.

Algocell and Zilion were developed by Mostafa “Neo” Mohsenvand. Zilion separates processor semantics from the world and runs batches of encounters on WebGPU. The npm 0.3.1 release provides Z80, NMOS 6502 and coherent-fetch 8086 cores. Historical Z80 0.2 is retained for comparison with earlier results; corrected 0.3 and the zilog-nmos variant are separate choices. BFF is not exported in this npm release.

This work builds on the primordial-soup experiments of Agüera y Arcas and colleagues, Alexander Mordvintsev's implementation, and subsequent Z80 work by Cicala and colleagues. Processor validation draws on the SingleStepTests suites and the project's analytic, differential and regression tests. Passing a processor test suite establishes specific tested behavior, not every property of the population model. See the source repository, paper registrations and exported manifests for the exact conditions.

Everything in this app runs locally. A saved checkpoint identifies the engine/world and stores a snapshot and continuation state for this app's scheduler. Historical snapshots and short illustrative browser runs should not be confused with the paper's archived replicated cohorts.

## Guide rewrite: outline and full draft

Outline: first encounter → run controls → presets → machine/world settings → lattice layers/inspection → measurements → culture/marker experiments → files/provenance → legacy controls and troubleshooting.

**Start with one question.** Select Historical stage G to watch a soup, or a seeded mechanism to inspect copying immediately. Read the preset explanation. Press Step to complete one update, or Run for a bounded burst. Pause keeps the current soup. Reset rebuilds the selected condition from its seed. Changing the machine or geometry begins a new run so histories cannot silently mix conditions.

**Read the world before comparing runs.** The seed controls this app's random initialization and scheduler. Width and height give the number of tapes. Each tape has L bytes; two tapes meet on a P-byte ring, which may have additional padding. A draw chooses an initiating tape and a neighbor; collisions reduce the number of active pairs. Budget counts processor steps, whose meaning depends on ISA (an 8086 REP iteration is one step). Mutation exponent k gives approximately pair draws / 2^k replacement attempts per update, not a fraction of all bytes flipped.

**Choose a processor deliberately.** Z80 0.2 retains historical core behavior. Z80 0.3 corrects it. The Zilog NMOS variant is a separate option. NMOS 6502 and coherent-fetch 8086 use different registers, stacks, flags and decoding. “First push at the last byte” and “stack top aliases the last byte” are not interchangeable. Random registers change encounter reset, while mutation changes tapes between encounters. The lethal-byte probability applies when that byte starts an instruction. Ablation skips selected decoded instructions according to the core's own decode contract; it does not delete bytes from memory.

**Look at the lattice.** Byte view gives stable colors to byte values and makes zero pale. Identity view groups equal complete tapes. Neither layer classifies heredity. Fetch view shows the last measured initiating encounter: open means instruction-stream fetching left A; confined means it did not; unknown means there is no current measurement. A partner may supply data even when fetching remains confined. Select a cell by pointer or cell number; its linear disassembly helps you read the bytes, but self-modification can make the executed path very different.

**Read measurements as measurements.** Zero fraction is the fraction of actual tape bytes equal to zero. Top-tape share is the fraction of cells carrying the most abundant exact byte string. Diversity is the number of distinct byte strings divided by the number of tapes (with entropy shown only if named explicitly). The chart's horizontal axis is updates, not wall-clock time or generations. A rise or fall does not by itself detect replication, confinement or inherited variation.

**Test a candidate.** Pause, select a tape and run the culture test. Each trial places that tape against a fresh random partner under the displayed world. Exact-copy frequency asks whether the offspring equals the selected sequence. Best cyclic similarity allows rotations and reports a different quantity. Open/confined results concern the test encounters. A short test samples a limited partner distribution and may miss rare behavior. For continued reproduction, transfer offspring to fresh partners; for heredity, plant a marker and score its carriage independently of survival.

**Try three experiments.** First, compare benign and lethal zero with the same seed/configuration and inspect both the zero chart and cultured tapes. Second, compare a pusher with a return closer and watch where instruction fetching goes; do not draw an ancestry arrow between them. Third, compare `04 5e ed b0` with a full-tape displacement and inspect the same payload marker in their offspring. The third experiment shows why confinement alone is insufficient: both can stay within A while treating variation differently.

**Save enough to explain a result.** Export a checkpoint before changing conditions. A new-format checkpoint carries bytes, configuration, initial-condition definition, counts, scheduler and RNG continuation, and machine identities. Save measurements separately as CSV; charts, fetch observations, assay results and the preset/source label are not part of the checkpoint. Export the manifest when citing an exact condition; a SHA-256 identifies content, not scientific validity. Legacy `.acell` files open in the legacy instrument and lack the information needed to promise exact old continuation. Files remain local. Invalid or incompatible imports should report the problem and preserve the existing run.

**Legacy controls.** The legacy instrument preserves square/hex rendering, 1×/2×/4×/8× speed, wheel/pinch zoom and drag pan, fit, averaged cells, Rainbow/Ocean/Thermal palettes, brightness/contrast/saturation and grid lines. Its settings include saved/overwritten/deleted user presets and Z80 suppression patterns. Its byte, exact-tape and diversity tabs are historical displays. Space toggles play, R resets, H opens help, Cmd/Ctrl-S saves, Cmd/Ctrl-O opens. Existing `.acell` drag-and-drop remains there. It is the historical optimized instrument, with its documented scheduler and measurement limitations.

**If execution is unavailable.** Use a WebGPU-enabled browser on localhost or HTTPS. Read the status message and retry initialization after correcting browser support. The interface, presets and documentation remain readable without a GPU. Run stops after its burst limit and while the page is hidden. No background server or account is required.

## Roadmap, risks and test strategy

Phase 1 (this branch): studied plan, new engine/world path and multi-ISA inspection; full source-bound preset catalogue; measured fetch overlay; fresh-partner culture; zero/top-share/diversity histories; modular runes state and components; About/Guide rewrite; responsive accessible design; versioned local exports; explicit legacy bridge. Validate and commit locally in logical steps.

Phase 2: move soup scheduling, measurements and rendering to interoperable resident GPU buffers after validating statistical and deterministic contracts; asynchronous readback/backpressure and long-run memory/performance profiling. Add paired-condition workspaces, batch cohorts, emergence censoring and statistical export. Expand decoder regression corpus where uncommon instructions remain conservative.

Phase 3: registered branching/serial culture protocol UI, ordered-write ancestry, multi-generation marker experiments and ecological competition control; match paper cohorts/provenance, not just world parameters. Add imported archived trajectories, cross-linked paper figures and replay verification; BFF only after an approved exported dependency supports it.

Risks: 8086 compile costs and software WebGPU support; random-state semantics differing across ISAs; confusion between archived contracts and new scheduler; static disassembly of self-modifying code; GPU resource leaks under rapid settings changes; checkpoints from incompatible versions; host gather/readback throughput and exact genotype counting. Mitigate by explicit contracts, independently seeded assays, transactional UI, disposal, validation and carefully bounded claims.

Checks: `npm run check`, `npm run lint`, `npm run build`. Playwright smoke covers load, every preset stepped briefly, each ISA and Z80 compatibility, inspector and culture, fetch layer, About and Guide, mobile layout, console errors and no external runtime calls. Scientific fixtures cover pusher copying, displacement payload retention/erasure, ablation masks, ring padding, decoder lengths, checkpoint continuation and malformed imports. Do not claim CPU fallback as GPU verification. GPU work remains in bursts ≤60 seconds. Capture desktop/mobile lattice, inspector, controls and documentation; critique actual screenshots and iterate. Record failures and exact final results below.

## Implementation and verification log

The initial plan was committed before implementation in `f99d394`; the dependency upgrade followed in `5ff2bfc`. All work is confined to `feat/app-zilion-0.3` in the requested worktree. No deployment or push was performed.

### Delivered architecture and behavior

- The main page is 624 lines, with five named components for experiment controls, the lattice, inspection, measurements and documentation. `session.svelte.ts` owns reactive lifecycle and transactional replacement; plain TypeScript modules own machine/world construction, deterministic population scheduling, scientific assays, disassembly, ablation, presets and checkpoint validation.
- Zilion 0.3.1 executes all three ISAs through `createEngine` and compiled worlds. Corrected Z80, frozen historical 0.2 and Zilog NMOS are explicit choices. Resident batch storage is reused and large batches are chunked. Exported manifests retain all four SHA-256 identities.
- Twenty-two presets cover every requested family, including all six registered lethality probabilities, both ring geometries, both offset competition chemistries and the paired displacement edit. Source paths, published seeds/horizons, findings, limitations and browser conditions are separate. Isolated transfer assays have no invented published lattice dimensions.
- The fetch layer distinguishes open, confined and unknown observations; mutation and later writes invalidate stale observations. Culture uses 16 matched trials through two generations, computes normalized gain against the original parent and displays exact copying, cyclic similarity and outside-parent fetch. The example exposes A/B output bytes, fetch/start maps, write counts, final registers, halt and trap status. Marker testing compares edited and unedited parents against matched fresh partners.
- Zero fraction, exact top-tape share and exact-string diversity are sampled each completed update, independent of panel visibility. The bounded chart history contains at most 180 observations. Export measurements as CSV. Checkpoints preserve exact continuation of this host scheduler, not chart/assay history.
- The UI starts paused, bounds a run to 55 seconds, pauses while hidden, and leaves its preview, controls and documentation readable when WebGPU is unavailable. Import errors preserve the current population. Settings and checkpoint replacement use generation guards and dispose superseded GPU resources.

### Deliberate boundaries and retained behavior

The original optimized instrument remains available at `/legacy`, including its square/hex rendering, palettes and brightness/contrast/saturation, speed control, local saved presets, historical Z80 step/trace inspection, opcode/species/diversity panels, keyboard/touch controls and `.acell`/PNG checkpoint workflow. The developer Z80 harness remains at `/dev/z80test`. These features were not silently discarded; they are intentionally kept behind an explicit legacy link while their replacements are designed. Legacy starts paused and also uses bounded bursts. KaTeX assets are local, eliminating the old CDN request. The legacy hex zero-fraction denominator and accessibility warnings were corrected. No requested feature was removed from the repository.

The first research desk uses CPU gathering, mutation, exact genotype census and Canvas2D rendering around GPU encounter execution. Its deterministic serial host scheduler and SplitMix initialization are newly identified contracts; archived GPU race order and NumPy competition planting are not recreated. A selected 0.2 core preserves processor behavior, not an entire archived soup trajectory. At 20,000 cells, a local browser observation completed about 100 updates in eight seconds; this is an interaction check, not a controlled throughput benchmark. The legacy display's roughly 500M instruction **slots**/second is not a comparable executed-instruction measurement.

Next-phase work remains: interoperable GPU-resident population scheduling/rendering and long-run profiling; paired conditions and replicated cohorts; the full registered branching/serial protocols and ordered-write ancestry; finer 8086 ModRM ablations; extended disassembler regression corpora; direct tape editing/planting and trajectory ancestry in the new UI; combined archives retaining preset provenance, measurements and assays; a public paper accession/figure browser; BFF after it is exported by an approved dependency. Static disassembly describes ordinary ISA byte lengths, not the altered fetch path under ablation or self-modification. Hex neighbor scheduling is supported in the new desk, but true hex rendering remains in legacy. The marker assay records serial byte-value retention, not causal ancestry.

### Verification and design review

On the completed source tree, `npm run check` passed with **0 errors and 0 warnings**, `npm run lint` passed, and `npm run build` generated the static site successfully. Official Svelte analysis was also run on the new components and rune module. `npm run test:e2e` passed **39/39 tests in 39.0 seconds** using installed Google Chrome with real WebGPU, one worker and a dedicated local server. All 22 presets were stepped; every ISA and the corrected NMOS Z80 were exercised. Fixtures verified frozen 0.2 equivalence against the legacy facade, chunk equivalence, register results, pusher/displacement behavior, lethal instruction-start semantics, exact no-copy masks, decoder lengths, padded stack geometry and continuation after restore. UI tests covered culture/marker controls, docs/focus, mobile overflow/anchors, no external requests, unavailable WebGPU and malformed-import recovery. A separate manual browser smoke ran and paused legacy for one second, opened settings and observed no page errors. GPU activity stayed in short bursts under 60 seconds.

The first complete test run found ambiguous accessible names on Compatibility/Variant and missing focus restoration when a dialog closed. Both were corrected; the final tests assert these behaviors. Screenshot review found that a broad mobile navigation selector hid the section links; it now targets only the legacy header link. The final mobile test checks both section anchors. Early tests also exposed the installed bundled Chromium's shader-compatibility limitation; the suite explicitly uses current Chrome. Zilion's published package produces missing-source-map warnings during direct module fixture imports; this does not affect execution. Legacy Mermaid/KaTeX remain a bundle-size cost to address separately.

Design critique led to centered dialogs, secondary text of at least 12px outside compact chart marks, neutral diversity coloring, stacked mobile charts, visible mobile section navigation, stable lattice geometry and focus-visible controls. Final desktop review shows a dominant lattice with controls and inspector separated by reading purpose, aligned percentage charts and restrained paper colors. Mobile keeps the soup first and offers direct links to the inspector and experiments. Long documentation uses an internal contents list and scrolling dialog; no horizontal overflow was observed at 390px. Remaining design work is comparative research workflows and denser long-run evidence, not hidden controls or ambiguous measurements.

Final reviewed screenshots in `docs/screenshots/`:

- `desktop-workspace.png`, `desktop-fetch.png`, `desktop-culture.png`: 1440px research desk, actual short-run histories and measured fetch layer.
- `culture-detail.png`, `culture-encounter.png`, `marker-detail.png`: result semantics and low-level outputs.
- `desktop-about.png`, `desktop-guide.png`: centered reading views.
- `mobile-overview.png`, `mobile-workspace.png`, `mobile-controls.png`, `mobile-inspector.png`, `mobile-about.png`, `mobile-guide.png`: 390px overview, full reading order, controls, inspection and documentation.
- `legacy-preserved.png`, `legacy-settings.png`: preserved historical instrument.

The images show new local runs and constructed mechanisms; none is presented as an archived paper result. Baseline screenshots remain alongside them for comparison.

### Source refinements agreed before implementation

- The detailed catalogue in `study-paper.md` includes all seeds, horizons, qualifiers and exact source paths. In padded segment worlds, Zilion's `lastByte` is the last **loaded segment byte**, not padding. Thus L12/P28 correctly anchors historical Z80 SP to B byte23, not ring byte27.
- The primary culture readout is the paper's two-generation normalized gain, not merely exact copying. Execute T against fresh partners, execute **every** first-generation offspring against new partners without success conditioning, and compare outputs with original T using best cyclic match. Average `(after-before)/(1-before)` over informative partners (before<0.75); gain≥0.3 is the operational screen. Display exact copies and ≥75% cyclic copies separately. All-noninformative samples are unknown, not sterile. n16 is a quick screen; n64 is the nominal paper panel size.
- The registered L32 displacement edit is `04→60` hex under P64 (96 mod64=32). The initial byte is also an instruction; arbitrary numerically convenient replacements need not preserve viable code. The seeded competition uses R=`04 5e ed b0` repeated and T=`a0 5e ed b0` plus independent random tails for each T cell, 50:50, not one cloned T tail.
- Historical random-register/dial streams have PCG hooks distinct from Zilion's new counter-based draws. Preserve these explicitly when a preset claims that condition; label the host scheduler separately. Corrected defaults and historical conditions cannot be merged under one seed label.
- The new data contract lives in `src/lib/lab/types.ts`. Checkpoints use `algocell.lab/1` and `serial-reflected-pcg/1`. All manifest identities are exported alongside runtime settings and host RNG state; hashes alone do not identify a whole population experiment.
- No runtime dependencies were added. Only the existing Zilion dependency changed, from `^0.2.0` to `^0.3.1`.
