# Algocell, one continuous instrument

Design specification recorded before UI implementation, 13 October 2026.
Base `6484a24`; branch `feat/app-unified`. The branch already existed at the
requested base with a clean worktree when this job began. No deployment branch
is involved. The preceding design, evolution screenshots and history since
`334e470` inform this evolution; the lattice, GPU residency and scientific
capabilities remain its foundation.

## Problem and decision

The application has accumulated an observation strip, transport, floating chart
widget, setup panel, probe dock, encounter inspector, species dialog and help
modal. Their locations imply separate instruments; their handlers often pause
or restart one shared population. The wordmark beside the Field view selector
is the most literal “Algocell box.” Selecting Open / confined pauses there.
The separate Processor selector also pauses and invokes `configure`, which
resets even when tape geometry is unchanged. These are UI policy decisions,
not a requirement of the GPU population.

The new composition is a single workbench: a persistent lattice viewport,
one right sidebar, a thin tool rail on the viewport's right edge, a restrained
top strip for looking, and one bottom transport/status strip. The sidebar is
the place to configure, investigate, learn and export. It never becomes a
second window. A selected tape stays marked on the lattice while its live
probe and encounter evidence occupy that same sidebar.

The alternative of retaining floating panels and simply matching their styles
was rejected: it would preserve the disconnected information architecture.
The alternative of a two-sided dashboard was rejected: the lattice deserves
the available area. The memorable visual is Algocell's woven byte field;
quiet charcoal chrome gives it a precise, instrument-like frame.

## Semantic model and proximity

| Domain              | Frequency                                    | Contents and placement                                                                                                                                                                                                                      |
| ------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| World               | Once per condition, occasional interventions | Core/ISA and compatibility, topology, lattice/tape/ring dimensions, seed, reset convention, chemistry, mutation, pairing and instruction budgets, ablation. World sidebar. Separate live controls from a staged next-world form.            |
| Run                 | Continually                                  | Play/pause, single update, speed, explicit restart. Stable bottom strip, beside run state and update rate.                                                                                                                                  |
| Look                | Continually                                  | Field view, byte palette, average/grid layers, brightness/contrast/saturation, zoom and fit. Compact viewport header, appearance disclosure in the same shell. No population changes.                                                       |
| Inspect and measure | Repeated exploratory loops                   | Probe address/bytes, encounter tests and disassembly, region snapshot and trend, marker intervention, sampled history, image comparison and population charts. Measure sidebar with contextual sections. Selection tools beside the canvas. |
| Experiments         | Beginning of a session, between trials       | Guided starts grouped by question, protocol, look-for text, source/limits and saved personal presets. Preview a condition, then explicitly start it.                                                                                        |
| Export              | At useful milestones                         | Population checkpoint/import, image, bounded timelapse, assay JSON. Export section within the sidebar; keyboard checkpoint access remains.                                                                                                  |
| Guide and About     | First visit or occasional reference          | Instructions, illustrated explanations, limits, shortcuts and project credits. Tabs in the same sidebar.                                                                                                                                    |

Probe navigation, marker position/value and encounter launch belong together.
Region selection, numeric corners, sample stamp and statistics belong together.
History capture, scrub/replay and return-live belong together; comparison and
recording share these visual artifacts. Population charts retain their existing
meanings and become a Measure section. Configuration that starts a condition
is kept out of these repeated observation loops.

## Users, tasks and measurement

Four realistic users guide the review:

- **Visitor from the README:** needs to see a live world immediately, understand
  a view, pause/step, and find a short explanation without leaving the field.
- **Researcher:** chooses a documented condition, verifies seed/core/geometry,
  changes one parameter, runs a protocol and saves the condition and evidence.
- **Power observer:** repeatedly selects neighbouring tapes, cultures candidates,
  follows marker value matches, samples regions and compares recorded views.
  The UI must not falsely imply that value matching measures causal lineage.
- **Teacher:** switches views and layers, zooms, steps, explains a selected tape,
  and returns to a running overview while presenting.

The repeatable task harness covers 12 end-to-end paths. Final exact paths and
observations are in `docs/review/unified/TASKS.md` and `scripts/unified-tasks.mjs`.

| Task                                                   | Main user            | Expected frequency |
| ------------------------------------------------------ | -------------------- | ------------------ |
| Pause, step and resume                                 | Visitor / teacher    | 10–30 per session  |
| Switch field views and return to bytes                 | All                  | 10–20 per session  |
| Change processor with existing tapes                   | Researcher           | 2–5 per session    |
| Edit seed/geometry and explicitly start a world        | Researcher           | 1–5 per session    |
| Choose and run a guided experiment                     | Visitor / researcher | 1–4 per session    |
| Select a cell, advance probe and open encounter bench  | Power observer       | 10–50 per session  |
| Run a culture and inspect fetched/start/write evidence | Power observer       | 5–20 per session   |
| Mark a byte and inspect matching cells                 | Power observer       | 2–10 per session   |
| Select/sample a region and examine statistics          | Researcher / teacher | 5–15 per session   |
| Record/scrub history and return live                   | Teacher / observer   | 2–8 per session    |
| Pin/reveal a reference and export evidence             | Researcher           | 1–5 per session    |
| Read Guide/About, return, change look and fit          | Visitor / teacher    | 1–10 per session   |

Measurements count actual scripted clicks, Euclidean pointer travel between
target centres in CSS pixels (fixed starting point), distinct semantic surfaces,
playing→paused transitions, world resets and engine object/device replacements.
Record pipeline rebuilds separately: compiling a new ISA or optional measurement
kernel is not replacing the engine or population. Keyboard paths are tested but
not mixed into pointer-path totals. Missing baseline functionality is reported,
not assigned an invented zero. Capture every meaningful panel/view/expanded
state under `before/`, and each review round under `after/round-N/`.

## One shell

```text
┌────────────────────────────────────────────────────────────┬──────────────┐
│ Algocell   [Field view ▾]   [Look]           [−] [+] [Fit]   │ sidebar tabs │
├────────────────────────────────────────────────────────┬───┼──────────────┤
│                                                        │ V │ World        │
│                                                        │ P │ live controls│
│                    continuous lattice                  │ R │              │
│                    + selected tape/region              │ M │ Next world   │
│                                                        │ C │ staged form  │
│                                                        │   │ explicit CTA │
├────────────────────────────────────────────────────────┴───┴──────────────┤
│ [Pause] [Step] [8×] [Restart] Running · updates/s · epoch · core · seed · L │
└───────────────────────────────────────────────────────────────────────────┘
```

Sidebar tabs: **World**, **Measure**, **Experiments**, **Guide**, **About**.
Export is a persistent sidebar section/action, not another window. Measure
subsections expose Probe/encounter, Region, Markers, History, Compare and Charts.
The rail contains **Select/pan (V), Probe (P), Region (R), Marker (M), Compare (C)**.
It controls interaction modes; Guide and configuration are never rail tools.
Tabs retain their selected sections, form drafts and evidence. The viewport and
engine component stay mounted. No duplicated play controls or old widget remain.

The status bar names run state, updates/s, epoch/update, core/compatibility,
seed, lattice size, tape length and chemistry. Status buttons jump to and focus
the corresponding control/section, expanding the sidebar as necessary. Values
describe the applied world, never an unconfirmed draft.

Opening the bench replaces the Measure content for the selected cell within the
sidebar. It keeps the probe address and return path visible. Live probe samples
refresh no more than twice per second; test results record the sampled parent,
condition and epoch rather than silently following subsequent mutations.
No encounter launch pauses the live population. Results are isolated assays.

## Continuity contract: explicit state machine

State is `(worldIdentity, playIntent, operation, draft, presentation)`.
`playIntent ∈ {playing, paused}`; `operation ∈ {idle, swapping, starting, importing}`.
Presentation includes tab, section, tool, view, camera, target and visual history.

| Event                                                    | Transition                                           | Population/engine invariant                                                                         |
| -------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Tab/section/help/shortcut sheet                          | Presentation only                                    | Same play intent, engine/device/buffers, bytes, epoch trajectory                                    |
| View/layer/tool/zoom/pan/probe target                    | Presentation only; bounded observation work          | Same play intent and world; optional measurement pipeline compilation allowed                       |
| Play/pause                                               | Toggle play intent                                   | Same world and camera                                                                               |
| Step                                                     | Submit one update when paused                        | No replacement; when playing keep playing                                                           |
| Speed, mutation, instruction/pair budget, ablation       | Apply subsequent-update parameter                    | No reset; same play intent                                                                          |
| Chemistry or compatible core change                      | idle→swapping→idle                                   | Drain/compile safely, reuse device and population buffers; retain bytes/counters/camera/play intent |
| Seed, lattice, topology, tape/ring/reset convention edit | Update draft only                                    | Applied world untouched; visible pending summary                                                    |
| Select experiment/personal preset                        | Stage condition and show protocol                    | Applied world untouched                                                                             |
| “Start new world with these settings”                    | idle→starting→idle, replace world content explicitly | One explicit reset; same engine object where supported; preserve play intent                        |
| Restart                                                  | Explicit reset of current applied condition          | Preserve play intent; never use draft implicitly                                                    |
| Region snapshot/marker/assay/export                      | Submit bounded operation on current condition        | Preserve play intent; snapshot has epoch; marker patches only intended byte                         |
| History scrub/replay/compare                             | Change displayed evidence                            | World keeps advancing if playing; live/recorded labels make this clear                              |
| Import checkpoint confirmation                           | idle→importing→idle                                  | Explicit world replacement, retain play intent                                                      |

All 256 byte values are representable by the current cores. Invalid/unsupported
instructions are execution semantics, not a reason to erase a same-length tape.
Core swaps choose valid ISA-specific reset/ablation/chemistry defaults and disclose
the changed semantics. Geometry changes cannot enter the live swap path.

Temporary queue drain/compilation may defer updates, but must not change the
play/pause state. Observation/assay busy flags must not stop unrelated simulation.
No component is keyed by epoch and no control is recreated by numeric ticks.
The existing local-GPU run limit is explicit in status/Guide if retained: it is
a resource policy, never a hidden effect of UI navigation. Hidden-tab suspension
must preserve play intent and resume on visibility; no silent world reset.

## Visual system

Retain Algocell's palette: ground `#0e0e0e`, panel `#151514`, raised controls
`#20201e`, text `#e8e4e0`, secondary `#b0a89e`, copper `#c8875a`.
The lattice's colour is data; byte palettes and fixed measurement scales remain.
Copper identifies active tools and intentional actions, not every border.

Use the existing system reading face, with system monospace only for addresses,
bytes and numeric data. Type scale: 11px metadata, 12px compact controls, 13px body,
16px section title, 20px sparse introductory title. Tabular numerals throughout
status and readouts. Sentence case labels. Align labels and values to a shared
grid; use 4/8/12/16/24px spacing. No gratuitous cards inside cards.

One control family: 32px desktop controls, 36–40px coarse-pointer targets;
4–6px corner radii, restrained one-pixel borders, consistent disabled opacity.
Segmented buttons for small enumerations. A custom accessible listbox for longer
choices with keyboard arrows, Enter/Space selection and Escape/outside close.
No native dropdown chrome. Sliders pair with readable values; numbers have
labelled bounds. Subtle 1px copper `:focus-visible` outline, no mouse-focus halo.

Panel expansion/selection uses 160ms ease-out colour/opacity/size responses,
respecting reduced motion. Never animate the lattice or remount controls on a
clock. Reserve widths for counters. Dense content scrolls within the sidebar;
section headers and primary navigation stay reachable.

Empty states teach the next action: “Select a tape on the lattice to inspect
its bytes”; “Record a few frames to compare changes”; “Drag a region or enter
its corners, then sample.” First run is already alive, with a concise World
description and a direct guided-experiment path, no onboarding overlay.

## Responsive layout

| Viewport        | Sidebar                                                    | Rail    | Canvas and bars                                                                      |
| --------------- | ---------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| 1280×800        | 352–368px                                                  | 40px    | Remaining width, 48px header, 40–44px status                                         |
| 1440×900        | 368–392px                                                  | 40px    | Larger uninterrupted field; same control spacing                                     |
| 1920×1080       | 392px maximum                                              | 40px    | Extra space belongs to the lattice                                                   |
| Narrow / mobile | Collapsed icon navigation; explicit expansion within shell | 36–40px | Bounded sidebar content; status prioritizes run and epoch; no horizontal page scroll |

The canvas measures its actual container using ResizeObserver. Sidebar resize
does not resize/reseed the world or automatically reset camera. Fit is explicit.
Headers, controls and selected tape highlights must not overlap; field overlays
(selection outline, history image, comparison) remain clipped to the viewport.

## Keyboard

Space toggles play; `.` steps; V/P/R/M/C select direct tools; 1–5 select the five
sidebar tabs; `?` opens the shortcut section in Guide; Escape dismisses a listbox
or returns the contextual tool to navigation; arrows move the selected probe.
Plus/minus zoom, F fits, Cmd/Ctrl+S exports checkpoint, Cmd/Ctrl+O imports.
No unmodified shortcuts fire in text/number/select/contenteditable controls.
Reset moves off the old R shortcut because R now selects Region; use an explicit
restart control (and a documented modified shortcut only if needed).

## Implementation and evidence gates

Keep GPUEngine and the original renderer authoritative. Reparent retained chart,
help, preset and scientific content rather than replacing capabilities. Refactor
LegacyInstrument into the shell; ObservationTools owns live observation state
and exposes shell snippets/props for rail/header/sidebar placement; EncounterBench
is in-flow. Shared custom controls prevent divergent styles. Avoid DOM portals,
imperative reparenting and independent fixed-position panels.

World draft data is distinct from applied settings. Live mutations read the
applied engine settings, so pending geometry cannot leak into a core swap.
Test hooks expose read-only identity and counters, with explicit user-equivalent
actions only where benchmark setup requires them. Tests count real reset/configure
calls and engine/device/population identities, not just a displayed “Running.”

Validation: Svelte MCP analysis, `npm run check`, `npm run lint`, production build,
and the complete Playwright suite (update superseded layout expectations while
retaining scientific assertions). Add continuity matrices, staged application,
live core-byte preservation, three viewport overlap/overflow checks, control DOM
mutation counts excluding numeric readouts, accessible custom-selector keyboard
tests, and absence of the author's prohibited word in visible/accessibility copy.

Freeze `6484a24` for reproducible before captures and paired benchmarks. Run the
existing engine harness for all five cores in compute/rendered mode and the real
app harness at native 8×. Use alternating paired order, short GPU bursts, idle
gaps, no concurrent GPU/check/build workloads. Report medians and paired changes;
investigate any core worse than −2%, retain all results and same-build controls
when noise prevents a clean claim. No population readback is added by default.

At least three review rounds each rerun task paths and the full state atlas.
Record screenshots, measured metrics, critique and fixes in
`docs/review/unified/ROUNDS.md`. Review with the demands of Figma's spatial
consistency, Linear's density/hierarchy, Blender's stable workspace, Ableton's
transport continuity and Observable's explicit evidence scope; these are design
criteria, not claims of equivalence or fabricated user A/B testing.

No new dependency is planned. Never stop the reviewer's 5175 server, leave GPU
jobs running, push, touch main, or commit local agent/configuration files.
