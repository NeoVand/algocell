# Unified application: baseline inventory

Baseline: `6484a24` (`feat/app-legacy-base`), before UI changes. Source line references below refer to that revision. This is a source-level inventory; measured pointer paths and new screenshots are separate review artifacts. `src/routes/+page.svelte` mounts one `LegacyInstrument`; the disconnection is within that component and its children, rather than separate application routes.

## What the “Algocell box” is

The most literal match is the **top-left observation strip**: `.observation-bar` contains the Algocell wordmark, a native **Field view** select, status/counter, zoom, a second row of tool buttons and a legend (`ObservationTools.svelte:667`). Its six views are Bytes, Tape identity, Open / confined, Zero density, Local diversity and Marker matches. It is not a core selector. Selecting Open / confined calls `onpause()` (`ObservationTools:150`); returning to another view does not restore playback. Entering or leaving confinement rebuilds GPU pipelines and clears the fetch-status observations (`gpu/engine.ts:667`), but does not reseed the lattice or reset its update count. The altered palette and newly unknown fetch observations can look like a reset.

A second possible interpretation is **Processor** in the separate Parameters panel. Its five core/compat options always call `chooseMachine` → `updateMachine(true)` → `engine.configure` (`LegacyInstrument:198–231`). `updateMachine` unconditionally writes `playing = false`, and `configure` calls `reset(settings.seed)` even when the new core uses the same tape geometry. The population really resets in this path, while selections, charts and history are cleared. The names of the two selectors and their positions do not explain this difference.

The old lower-left **measurement widget** is a third independently positioned surface. It contains an instruction-slot rate, mutation display and the Opcodes/Species/Diversity charts. It is not the simulation owner, although it visually resembles a separate instrument.

## Lifecycle terms and disruption key

- **P**: the UI sets playback to paused and does not automatically restore it.
- **R**: population reseeded/replaced, update count reset, measurements cleared.
- **B**: all population GPU buffers destroyed and re-created, with camera reset.
- **C**: compute/render pipelines rebuilt, without creating another `GPUEngine` object.
- **I**: an isolated culture/assay machine is created on demand; it is not the live population engine.
- **—**: none of the preceding; normal presentation changes preserve the live population.

`GPUEngine` itself is constructed once in `onMount` (`LegacyInstrument:632`), destroyed on unmount (`:668`), and is not replaced by any ordinary baseline control. `setMachine` (`gpu/engine.ts:598`) detects changes to width, height, topology, tape length or ring length: those call `changeGridConfig` (`:711`), which destroys/rebuilds buffers and pipelines, initializes soup and resets camera/count. Otherwise `setMachine` recompiles pipelines while preserving the soup. `configure` additionally resets and optionally writes a tiled/competition population (`:636`). “Engine re-creation” must therefore be measured separately from object identity, buffer recreation and pipeline compilation.

The animation loop temporarily stops submitting simulation work while `machineBusy`, `observationBusy`, `checkpointBusy` or `cultureBusy` is true (`LegacyInstrument:697`); this can interrupt advancement even when `playing` remains true. Independently, **all baseline runs auto-pause after 55 seconds and when the document is hidden** (`:696`, `:2147`). No control resumes an implicitly paused run except Play/Space.

## Surfaces and all interactive controls

Repeated generated controls (one per preset, opcode, tape byte or help section) are listed as families with their complete option sets below. Native tooltips and browser file/download surfaces are distinguished from app panels.

### Canvas and original transport

| Control or surface                         | Location / source                                       | Purpose                                                                     | Effect                                                      |
| ------------------------------------------ | ------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Full-screen lattice canvas                 | `LegacyInstrument:2162–2195`                            | Square or hexagonal byte lattice, rendered by the existing GPU shader       | —                                                           |
| Pointer hover / tape tooltip               | Canvas; `:977–1050`, `:4087–4125`                       | Highlight cell; read one tape, show mnemonic/operand cells; at most twice/s | —                                                           |
| Left/middle drag; one-finger drag          | Canvas; `:1053`, `:1113`                                | Pan camera                                                                  | —                                                           |
| Wheel / trackpad pinch; touch pinch        | Canvas; `:1090`, `:1136`                                | Zoom around pointer/gesture                                                 | —                                                           |
| Single click/tap                           | Canvas; `:1070`, `:1186`                                | Select probe after click delay; opens probe dock                            | —                                                           |
| Double click/double tap; Fit               | Canvas; `:1241`                                         | Reset camera, close selected tape inspector                                 | —                                                           |
| Play / Pause                               | Top-right `.toolbar`; `:2223`                           | Toggle run, renew 55-second run deadline                                    | P only by explicit pause                                    |
| Speed popup and 1×, 2×, 4×, 8×             | Transport; `:2244–2263`                                 | Updates requested per animation frame                                       | —                                                           |
| Reset                                      | Transport; `:2266`, handler `:1247`                     | Recreate initial population from current seed/initial condition             | R; tiled/competition also P+C; random reset retains playing |
| Save checkpoint                            | Transport; `:2283`, handler `:1280`                     | Save PNG-compatible `.acell` with settings and population bytes             | P; small exclusive readback interval; no R/B                |
| Load checkpoint / hidden input / file drop | Transport, canvas; `:2304`, `:1339`, `:1382`            | Read local `.acell`, configure condition and replace soup                   | P+R+C; B if geometry differs                                |
| Settings open/close                        | Transport → floating Parameters panel; `:2328`, `:3022` | Show/hide configuration                                                     | —                                                           |
| Help open/close                            | Transport → large Help dialog; `:2351`, `:4382`         | Show/hide documentation                                                     | —                                                           |
| Collapse / expand toolbar                  | Transport; `:2373`                                      | Hide transport buttons, speed popup and Parameters panel                    | —                                                           |
| WebGPU unavailable message                 | Canvas-level error layer; `:2162`                       | Explains unsupported browser; documentation remains available               | No engine initialized                                       |

### Observation strip, views and tool dock

| Control or surface                                             | Location / source in `ObservationTools.svelte`              | Purpose                                                                 | Effect                                           |
| -------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------ |
| Field view: Bytes                                              | Top-left select, `:677`; mode list `:41`                    | Inherited byte palette, individual bytes when zoomed                    | —; C when leaving confinement                    |
| Tape identity                                                  | Same select                                                 | Stable full-tape hash fingerprints; possible colour collisions          | —; C when leaving confinement                    |
| Open / confined                                                | Same select                                                 | Last initiating encounter: unknown/no exit/outside A, including padding | **P+C** entering; no soup reset                  |
| Zero density                                                   | Same select                                                 | Fraction of zero bytes in tape, fixed 0–100% scale                      | —; C when leaving confinement                    |
| Local diversity                                                | Same select                                                 | Exact distinct tapes / valid cells in 3×3 coordinate neighborhood       | —; C when leaving confinement                    |
| Marker matches                                                 | Same select                                                 | Same-position byte matches, not ancestry                                | —; C when leaving confinement                    |
| Running/Paused + update count                                  | Observation heading `:686`                                  | Noninteractive run-state readout                                        | —                                                |
| Zoom out / Zoom in / Fit                                       | Heading `:690`                                              | Same camera as canvas gestures                                          | —                                                |
| Navigate / Probe / Region / History / Compare / Record         | Top-left tool tabs `:698`; mobile native tool select `:707` | Select one contextual dock                                              | Region **P**; others —                           |
| Field legend                                                   | Strip `:714`                                                | Per-mode scale, scope, collision/ancestry limitations                   | —                                                |
| Contextual dock close                                          | Lower-right desktop / bottom mobile `:767`                  | Return to Navigate; exit history/compare; marker view returns to Bytes  | — (C if exiting confinement through mode change) |
| Probe cell index / Probe / Previous / Next                     | Probe dock `:780–800`                                       | Select one bounded tape by index                                        | —                                                |
| Center probe                                                   | Dock header `:758`, `:242`                                  | Pan selected cell into unobscured canvas                                | —                                                |
| Probe byte tape + coordinates/core/update/zero fraction/length | Probe dock `:801–812`                                       | Live tape readback at 2Hz while running                                 | —                                                |
| Open encounter bench                                           | Probe dock `:814` → separate Tape inspector dialog          | Snapshot tape and show detailed assays                                  | **P** twice through callbacks; no R              |
| Region from/to x/y (four inputs)                               | Region dock `:830–866`                                      | Define rectangular region; editing invalidates held sample              | — (tool activation already P)                    |
| Region drag / touch drag                                       | Canvas while Region selected, `:328`, `:581`                | Outline region and measure at release                                   | P at drag start + measurement                    |
| Measure region                                                 | Region dock `:868`; implementation `:344`                   | Exact sample and add one bounded trend point                            | **P**, population readback, no R/B/C             |
| Frame region                                                   | Dock header `:758`, `:242`                                  | Zoom/pan region into visible canvas                                     | —                                                |
| Record recent frames checkbox                                  | History dock `:930`                                         | Opt-in visual frame capture, 1Hz while playing; retain 24               | —                                                |
| History preview / frame scrubber                               | History dock `:940–954`                                     | Show recorded image in large canvas overlay; world not rewound          | **P** when scrubbing                             |
| Replay / Stop replay                                           | History dock `:958`                                         | Cycle sampled frames every 400ms; reduced-motion respected              | **P** when replay starts                         |
| Return live                                                    | History dock `:970`                                         | Remove historical overlay; **does not resume playback**                 | —                                                |
| Save frame                                                     | History dock `:971`                                         | PNG of selected sampled frame                                           | —                                                |
| Clear history                                                  | History dock `:971`                                         | Discard visual history and return live view                             | —                                                |
| Pin / Replace reference                                        | Compare dock `:986`                                         | Capture one full-resolution image, condition/mode/update/camera         | —                                                |
| Show comparison checkbox                                       | Compare dock `:999`                                         | Overlay reference to left of split; live view on right                  | —                                                |
| Reveal reference slider                                        | Compare dock `:1002`                                        | Set split 5–95%; camera mismatch warning                                | —                                                |
| Save PNG snapshot                                              | Record dock `:1014`                                         | Render and export actual current field image                            | —                                                |
| Export WebM                                                    | Record dock `:1015`                                         | Browser MediaRecorder export of sampled frames at 5 fps                 | —                                                |
| Start recording history                                        | Empty Record dock `:1021`                                   | Switch to History and enable capture                                    | —                                                |
| Follow a byte value disclosure                                 | Probe dock or marker mode `:1039`                           | Expose marker controls                                                  | —                                                |
| Marker Position / Value                                        | Marker disclosure `:1042`                                   | Set matching byte position (0…L−1), value (0…255)                       | —                                                |
| Show matches                                                   | Marker disclosure `:1059`                                   | Apply marker selector and show Marker matches                           | —; C if leaving confinement                      |
| Plant in cell                                                  | Marker disclosure `:1064`, implementation `:385`            | Mutate one chosen byte, refresh probe and show matches                  | **P**, one cell write; no whole-world R          |
| Probe/region outline, selected-cell label                      | Canvas-linked overlays `:637–666`                           | Tie dock data to lattice location                                       | —                                                |
| History/reference image overlays                               | Canvas layer `:637–666`                                     | Visual evidence alongside current canvas                                | —                                                |
| Errors/notices                                                 | Dock footer `:1076`                                         | Explain explicit actions and failures                                   | —                                                |

### Old measurement widget, tape and species inspectors

| Control or measurement                       | Location / source in `LegacyInstrument.svelte` | Purpose                                                                                                   | Effect                                                                    |
| -------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| ops/s readout                                | Lower-left widget `:2391`                      | Proposed instruction-slot budget of completed GPU work, including halted slots; not executed instructions | —                                                                         |
| Mutation readout                             | Same widget                                    | 1/2^k or off                                                                                              | —                                                                         |
| Toggle chart                                 | Widget `:2408`                                 | Expand/collapse chart area                                                                                | —                                                                         |
| Opcodes / Species / Diversity tabs           | Widget `:2427–2449`                            | Select population chart                                                                                   | —; Species/Diversity enables hash/tape analysis                           |
| Opcode trends, ranked mnemonic tiles         | Widget `:2450–2540`                            | Top inherited-byte counts/concentration, hover linked chart                                               | —                                                                         |
| Click/Enter/Space opcode tile                | Widget `:2483`                                 | Toggle suppression of that byte opcode                                                                    | Live ablation; no P/R/B/C                                                 |
| Opcode tooltip                               | Floating pointer tooltip `:2969–2993`          | Rank, mnemonic, hex, concentration/frequency and suppression instruction                                  | —                                                                         |
| Suppression chips / remove                   | All chart tabs `:2517`, `:2621`, `:2714`       | Remove an ablation pattern, same settings as Parameters                                                   | Live; no P/R/B/C                                                          |
| Species trends / hover / abundance tiles     | Widget `:2541–2644`                            | Full-tape content-hash group shares, top groups                                                           | —                                                                         |
| Species tile click/Enter/Space               | Widget `:2602` → separate Species modal        | Inspect frozen exemplar, fraction, population count, hash                                                 | — (unlike Open encounter bench, opening this modal does not itself pause) |
| Diversity chart and values                   | Widget `:2645–2738`                            | Richness, Shannon entropy/effective diversity, Simpson diversity, top share; hash groups                  | —                                                                         |
| Tape inspector / close / backdrop dismiss    | Separate right dialog `:2770–2818`             | Selected cell’s frozen byte grid, static disassembly, bench                                               | Opening P; close —                                                        |
| Disassembly disclosure                       | Tape/species inspector snippet `:2740`         | Address, bytes and selected ISA mnemonic                                                                  | —                                                                         |
| Species inspector / close / backdrop dismiss | Separate modal `:2820–2941`                    | Exemplar bytes (hex layout for L19), share/count/hash and bench                                           | —                                                                         |
| Species byte hover tooltip                   | Modal grid + floating tooltip `:2943–2967`     | Hexadecimal/mnemonic detail for exemplar byte                                                             | —                                                                         |

These charts measure composition, not lineage. Causal lineage recording is absent. Marker matches and marker-transfer assays are the existing supported route for following a changed byte; they must not be relabelled as proven ancestry.

### Encounter bench

`EncounterBench.svelte:1–354`; mounted inside either tape or species inspector through `{#key tape}` (`LegacyInstrument:2754`). Changing the tape destroys/remounts bench state. The tape is frozen and the initial settings are captured by `untrack`, so a changed condition produces “close the bench and inspect again” rather than staying live.

| Control / measurement                                                     | Purpose                                                                                 | Effect                                                                 |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Culture / Self-pair / Partner sensitivity / Marker transfer tabs (`:106`) | Select independent question/result                                                      | —                                                                      |
| Partners native select: 16, 32, 64 (`:126`)                               | Trial count; seed displayed, mutation off                                               | —                                                                      |
| Run culture test / Run comparison tests (`:136`, run `:55`)               | Two generations versus fresh partners, or matched self/partner/marker assays            | **P+I**; main soup retained, main simulation blocked while cultureBusy |
| Marker intervention disclosure (`:145`)                                   | Assay-local intervention settings                                                       | —                                                                      |
| Byte position / New value (`:149`)                                        | Perturb selected byte for paired assays, show old→new                                   | —; does not plant live population                                      |
| Culture result table (`:192`)                                             | Transfer number, gain, exact/rotated copy counts, outside-A counts; confinement summary | —                                                                      |
| Self-pair result (`:210`)                                                 | Changed positions in A/B; unchanged is not proof of copying                             | —                                                                      |
| Partner sensitivity result (`:236`)                                       | Changed offspring/parents and mean changed B positions                                  | —                                                                      |
| Original partner / Changed partner (`:254`)                               | Switch matched encounter record                                                         | —                                                                      |
| Marker retention bars (`:265`)                                            | Changed parent vs original control across transfers, 0–100%                             | —                                                                      |
| Fetched bytes / Instruction starts / Write activity (`:295`)              | Measured fetched addresses, starts and A/B/padding write counts                         | —                                                                      |
| Example steps / stopping reason (`:289`)                                  | Budget, halt/trap outcome                                                               | —                                                                      |
| Byte tooltips in A/B/padding lanes (`:324`)                               | Address, before/after, nonretained padding                                              | —                                                                      |
| Partner after encounter disclosure (`:343`)                               | Result tape; changed bytes distinguished from write counts                              | —                                                                      |
| Save evidence JSON (`:350`)                                               | Export actual result/settings/manifest                                                  | —                                                                      |

### Parameters, presets and appearance

| Control                                                 | Location / source in `LegacyInstrument.svelte`                               | Purpose                                                                       | Effect                                                                                                             |
| ------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Saved preset chips                                      | Parameters `:3198`; `applyPreset:1645`                                       | Apply stored subset/full configuration                                        | Highest disruption among differences; world/seed P+R+C (+B geometry), live machine settings P+C, pure appearance — |
| Save settings toggle / name / Save                      | Parameters `:3094–3137`                                                      | Persist user preset in local storage                                          | —                                                                                                                  |
| Delete selected preset                                  | Parameters `:3074`                                                           | Remove selected matching local preset (including seeded built-ins)            | —                                                                                                                  |
| Preset overwrite dialog: Cancel / Overwrite / backdrop  | Separate dialog `:4060–4085`                                                 | Confirm replacing same case-insensitive preset name                           | —                                                                                                                  |
| Experiments native selector                             | Parameters `:3141`                                                           | Apply one of 22 registered/guided configurations immediately                  | **P+R+C**, B if geometry changes                                                                                   |
| Finding, limits & sources disclosure                    | Experiment description `:3166`                                               | Provenance, condition, registered horizon/reset, observations and limitations | —                                                                                                                  |
| Processor native selector                               | Parameters `:3230`                                                           | Z80 compat 0.2, Z80 0.3 classic, Z80 0.3 Zilog NMOS, NMOS 6502, coherent 8086 | **P+R+C**, B only if geometry differs                                                                              |
| World contract disclosure                               | Parameters `:3243`                                                           | Reveal tape/ring/reset contract controls                                      | —                                                                                                                  |
| Tape L 4–32                                             | Contract `:3249`                                                             | Set tape length; raises minimum ring; random initial condition                | **P+R+B+C**                                                                                                        |
| Ring P 2L–128                                           | Contract `:3265`                                                             | Set execution ring length                                                     | **P+R+B+C**                                                                                                        |
| Stack placement native selector                         | Contract `:3283`                                                             | Top at B’s last byte / First push at B’s last byte / Original SP=FFFF (Z80)   | **P+R+C**                                                                                                          |
| Random registers checkbox                               | Contract `:3297`                                                             | Alter encounter reset registers                                               | **P+C**, soup retained but charts/selection/history cleared                                                        |
| Dial stream native selector                             | Contract `:3307`                                                             | Historical PCG / Counter                                                      | **P+C**, same measurement clearing                                                                                 |
| Lethal zero slider 0–100%                               | Parameters `:3328`                                                           | Probability that zero instruction starts halt                                 | **P+C**, same measurement clearing                                                                                 |
| Square / Hex grid buttons                               | Parameters `:3391`, `:3416`                                                  | Switch topology, set L16/P32 or L19/P38, recompute viewport-fit dimensions    | **P+R+B+C**                                                                                                        |
| W / H number fields                                     | Parameters `:3435`, `:3447`                                                  | Stage width/height 50–500; state already differs from applied engine          | — until Apply or another updateMachine action accidentally applies values                                          |
| Apply & Reset button                                    | Grid row `:3456`                                                             | Apply width/height                                                            | **P+R+B+C** when changed                                                                                           |
| Seed number / Reset with seed                           | Parameters `:3524`, `:3532`                                                  | Stage seed; explicit reset                                                    | Input —; reset R (P+C for nonrandom start)                                                                         |
| Mutation slider / Mutation off checkbox                 | Parameters `:3588`, `:3606`                                                  | Live per-byte mutation exponent; off k≥31                                     | —                                                                                                                  |
| Pairs/batch slider 256…8192                             | Parameters `:3663`                                                           | Live proposed pair budget                                                     | —                                                                                                                  |
| Execution steps slider 16…1024                          | Parameters `:3712`                                                           | Live encounter instruction-slot budget                                        | —                                                                                                                  |
| Suppress pattern input / Enter / preview                | Parameters `:3788`; handler `:1426`                                          | Validated mnemonic/family/exact opcode, semicolon lists; preview match count  | Live masks; no P/R/B/C                                                                                             |
| Suppression chips remove / Clear all                    | Parameters `:3814`, `:3774`                                                  | Remove one/all active patterns                                                | Live masks; no P/R/B/C                                                                                             |
| Rainbow / Ocean / Thermal                               | Appearance `:3897`; `colormap.ts:319`                                        | Byte-value palette                                                            | —                                                                                                                  |
| Color adjustments disclosure                            | Appearance `:3873`                                                           | Reveal three sliders                                                          | —                                                                                                                  |
| Brightness −0.5…0.5 / Contrast 0.3…2.5 / Saturation 0…3 | Appearance `:3911–3940`                                                      | Live renderer adjustment                                                      | —                                                                                                                  |
| Color Reset                                             | Appearance `:3942`                                                           | Restore 0/1/1 adjustments                                                     | —                                                                                                                  |
| Grid lines checkbox                                     | Appearance `:3953`                                                           | Renderer grid overlay                                                         | —                                                                                                                  |
| Detail: Detailed / Simple                               | Appearance `:4011`, `:4036`                                                  | Show bytes versus averaged tape colour                                        | —                                                                                                                  |
| Question-mark explanatory popovers                      | Presets, Grid, Seed, Mutation, Pairs, Steps, Suppression, Appearance, Detail | Hover or click explanatory text                                               | —                                                                                                                  |

The four built-in editable local presets (`src/lib/presets.ts`) are **Classic**, **Hex organic**, **No block copy**, **No-copy**. Legacy presets lacking a `machine` field restore historical Z80/reset semantics, which can unexpectedly elevate an apparently small suppression change into a reset when another core is active.

The 22 experiment options (`src/lib/lab/presets.ts:88–493`) are:

- Discovery: **Random code · benign zero**, **Random code · corrected Z80**, **Random code · NMOS 6502**, **Random code · coherent 8086**.
- Intervention: **Make zero lethal**; **Lethality dial · 0%, 1%, 3%, 10%, 30%, 100%**; **Random registers**; **Ring geometry · 12 on 24**, **Ring geometry · 12 on 28**; **No block copy**, **No-copy ablation**.
- Demonstration: **Competing copiers · benign zero**, **Competing copiers · lethal zero**, **Short-prefix copier**, **Full-tape copier**, **Load and push**, **Return-driven copier**.

Every option supplies initial bytes/condition, core, geometry, chemistry, budgets, mutation and reset semantics; descriptions distinguish explanatory demonstrations from registered cohort conditions. Internal historical identifiers must remain compatible without being displayed as UI language.

### Guide, About and keyboard

Help is a large separate `.modal` (`LegacyInstrument:4127–5647`) with **Overview, Guide, Visuals, Z80, Params, Analysis, Keys, About** tabs. Tab changes and all close/backdrop actions preserve playback. Overview and Visuals retain illustrated/Mermaid explanations; Z80 provides the instruction-machine explanation; Params explains settings; Analysis explains population summaries. Guide’s sections are Choose one question; Run/pause/reset; Set machine/geometry; Make zero dangerous; Read six field views; Probe/region; Encounter; Culture/marker transfer; History/comparison/recording; Keep condition/evidence; Support/shortcuts (`src/lib/lab/content.ts`). About covers copying versus inheriting, test scopes, machine/world, and people/source credits, including outbound source links. These documentation surfaces are also available without WebGPU.

| Shortcut / gesture | Scope and behavior                                                                                                                         | Effect                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| Space              | Body focused only: Play/Pause                                                                                                              | Explicit P toggle               |
| R                  | Body focused only: Reset                                                                                                                   | Same as reset control           |
| H                  | Body focused only: Help                                                                                                                    | —                               |
| S                  | Body focused only: Toggle original chart                                                                                                   | —                               |
| F                  | Body focused only: Fit camera                                                                                                              | —                               |
| Cmd/Ctrl+S         | Body focused only: Save checkpoint, prevents browser save                                                                                  | P                               |
| Cmd/Ctrl+O         | Body focused only: Open checkpoint chooser                                                                                                 | Load result P+R+C (+B geometry) |
| Escape             | Close settings/help/inspectors/overwrite; ObservationTools returns Navigate and exits history/compare; body branch also closes speed popup | —; does not resume              |
| Arrow keys         | Probe tool active, except text/select/textarea: move selected cell ±1 or ±width                                                            | —                               |
| Enter / Space      | Focused opcode/species tile: suppress opcode / inspect exemplar                                                                            | Live mask / separate inspector  |
| Enter              | Preset-name input saves; suppression input applies pattern                                                                                 | — / live mask                   |
| Escape             | Preset-name input closes save row                                                                                                          | —                               |
| Tab / Shift+Tab    | Native focus traversal; native selects own keyboard interactions                                                                           | —                               |

No baseline Step button or `.` shortcut, direct tool hotkeys, sidebar-tab hotkeys, or `?` shortcut sheet exists. The root handler requires `e.target === document.body`, so many shortcuts silently stop working after focusing a button. Native select menus and the file chooser are OS/browser surfaces; the four selectors most likely to be seen are Field view, Processor, Experiments and Test partners.

## Why it feels disconnected

1. Run lives upper-right, Look upper-left, measurements lower-left, configuration in a separate right panel, probes lower-right, encounters in another dialog, help in a large centered dialog. Operations used together require travel across the viewport.
2. The probe is live, but opening its bench pauses and freezes a snapshot. Species inspection has different pause behavior. Closing either does not restore the previous context consistently.
3. UI state and world state are mixed in shared handlers. `clearMeasurements` resets observation selections/history even for same-population chemistry changes (`LegacyInstrument:1257`). `ObservationTools.syncSettings` also treats any settings difference as reason to `resetSelection` (`:181`), so even live mutation/budget changes discard the probe and recorded frames.
4. Width/height/seed fields are nominally staged, but values are shared with `settings()`. A later unrelated `updateMachine` can apply an unfinished geometry edit. Other destructive controls apply immediately.
5. Wordmark-adjacent display selector looks like an application switch. The Processor selector does not advertise its forced reseed. Confinement view silently pauses even though it is a visualization.
6. Original chart tiles also change instruction ablation; the current action is disclosed only by tooltip. Identical patterns appear in both Parameters and the measurement widget.
7. There is no causal lineage recorder; marker matching is the honest available measurement. Documentation and inspectors must retain that distinction.

## Markup extraction map for implementation

All ranges refer to baseline `LegacyInstrument.svelte`; move blocks by semantic boundaries rather than assuming unchanged line numbers after edits.

| Baseline lines | Block / migration destination                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------- |
| 1–2082         | State, engine lifecycle, measurement math, pointer/setting handlers; retain GPU owner             |
| 2084–2142      | Global keyboard; replace with centralized shell shortcuts                                         |
| 2147–2151      | Visibility pause                                                                                  |
| 2152–2161      | Root and checkpoint file input                                                                    |
| 2162–2195      | Canvas container and canvas; keep mounted once                                                    |
| 2197–2217      | ObservationTools binding; recompose rail + sidebar tools                                          |
| 2219–2389      | Original transport → persistent run strip                                                         |
| 2391–2738      | Original measurement widget → Measure/Charts section                                              |
| 2740–2750      | Disassembly snippet → sidebar inspector                                                           |
| 2752–2768      | EncounterBench snippet; remove keyed-by-live-tape remounting                                      |
| 2770–2818      | Tape inspector wrapper → sidebar content                                                          |
| 2820–2941      | Species inspector wrapper → sidebar content                                                       |
| 2943–2967      | Species byte tooltip                                                                              |
| 2969–2993      | Byte-chart tooltip                                                                                |
| 2995–4058      | Parameters wrapper → World/Look/Experiments sections                                              |
| 3037–3224      | Local presets and experiment selector/description                                                 |
| 3226–3344      | Processor, world contract and chemistry                                                           |
| 3346–3545      | Grid and seed                                                                                     |
| 3547–3730      | Mutation and budgets                                                                              |
| 3732–3835      | Ablation                                                                                          |
| 3837–3970      | Appearance                                                                                        |
| 3972–4055      | Detail/averaging                                                                                  |
| 4060–4085      | Preset overwrite → inline confirmation                                                            |
| 4087–4125      | Hover tape visualization                                                                          |
| 4127–4394      | Help modal wrapper/tabs → Guide/About tabs                                                        |
| 4395–4401      | Guide content renderer                                                                            |
| 4402–4962      | Overview illustrated content                                                                      |
| 4963–5285      | Visuals illustrated content                                                                       |
| 5286–5467      | Z80 content                                                                                       |
| 5468–5531      | Parameters explanation                                                                            |
| 5532–5600      | Analysis explanation                                                                              |
| 5601–5617      | Shortcut list                                                                                     |
| 5618–5647      | About content                                                                                     |
| 5651–7861      | Existing scoped visual system; preserve byte-grid/diagram rules, replace disconnected positioning |

## Previous design and implementation history

Read `docs/DESIGN_EVOLUTION.md` and `git log 334e470..6484a24`. The earlier spec deliberately retained floating edge instruments, called native selects acceptable, paused region/history work and explicitly defined core changes as fresh conditions. The current behavior is therefore partly a shipped design decision, not merely an accidental regression. The new unified brief supersedes those decisions while retaining lattice identity and scientific meanings.

History since `334e470`: `8378044` restored the original app route; `6457fda` fixed geometry fitting; `ee67698` validated GPU equivalence/performance; `1ba8712` committed the evolution plan; `91656de` baseline evidence; `fd58c89` GPU modes/probes/assays; `dcb2b87` experiment names/workflow; `0d90aa0` observation/encounter UI; `70184f1` linked-tool polish; `02eb485` legends/framing; `7227f8f` mobile marker bounds; `59cd038` mobile sticky header; `80f961d` reproducible performance/capture controls; `6484a24` review evidence.

Representative source screenshots inspected: `mode-bytes-1440.png`, `tool-probe-1440.png`, `bench-culture-1440.png`. They confirm the four-corner separation and the bench replacing the probe context while the original chrome remains. The previous review’s complete screenshot index is `docs/review/evolution/SCREENSHOTS.md` (58 view/tool captures at desktop/mobile); this task’s fresh baseline capture belongs under `before/`.

The Svelte code-writer and core-best-practices skills were read, and official `$effect` / lifecycle documentation fetched. Baseline `ObservationTools.svelte` passed Svelte MCP autofixer with no issues or suggestions. This inventory changes documentation only.
