# Algocell: an instrument for looking, testing and keeping evidence

Design recorded before implementation, 13 October 2026. Base: `ee67698` on
`feat/app-legacy-base`. This document is the implementation brief and the record of
the design critique. The public branch is outside this work.

## 1. What the original gets right

The subject is the lattice. At full extent it is a finely woven multicolour field;
at close range the same field resolves into individual inherited bytes. There is
no conceptual jump between overview and detail. Square tapes use little mosaics;
hex tapes have actual hexagonal neighbourhoods and concentric byte geometry.
Resolution-aware averaging keeps far-away tapes legible. Wheel/pinch zoom, drag
pan and fit make exploration feel direct. These are the instrument's identity,
not decoration to replace with a dashboard.

The chrome floats over the field: a compact rounded toolbar at the upper right,
a small measurement instrument at the lower left, narrow settings, and quiet dark
overlays. It occupies the edges and leaves the middle open. Translucency, subtle
borders and modest radii separate controls from a very busy image without making
the canvas feel like a thumbnail. The original screenshots in
`docs/review/screenshots/app-1440-main.png` and `app-390-main.png` establish this
base. The restored review reports pixel identity for the core field and toolbar.

Colour tokens to retain:

| Role           | Value                | Use                                 |
| -------------- | -------------------- | ----------------------------------- |
| Ground         | `#0e0e0e`            | Unoccupied viewport                 |
| Instrument     | `rgba(18,18,18,.94)` | Floating controls                   |
| Raised surface | `rgba(24,24,24,.98)` | Menus and inspectors                |
| Main text      | `#e8e4e0`            | Reading and important values        |
| Secondary text | `#b0a89e`            | Supporting labels                   |
| Copper         | `#c8875a`            | Active tools, focus, selected state |

Byte colour is data. Retain Rainbow, Ocean and Thermal and the existing
brightness/contrast/saturation controls. Measurement scales must have stable
endpoints and named meanings; their values must not shift when an appearance
slider changes. Use cool blue for no observed departure, amber for departure,
and neutral grey for unknown. Shape/text accompany colours.

Keep the existing system monospace face for addresses, hexadecimal bytes and
tabular measurements, and the existing reading face for help. The compact
11–13px data typography belongs here, but explanatory text should be 12–14px and
help 14–16px. Avoid a new display face, hero headline, badge system or generic
card grid. Existing uppercase micro-labels are heritage, not a pattern to spread.

The original has unusually useful detail: mnemonic/operand distinction in the
hover tape, opcode suppression chips, real byte geometry, camera fit, keyboard
shortcuts, and a checkpoint that is also a viewable image. Preserve these.

Its weaknesses are connections: observation modes are buried in settings; a
modal hides the thing being inspected; chart scales are implicit; the selected
condition is hard to see; hovering reads the whole soup. The restoration added
machine features but did not give them an observation workflow.

## 2. Capabilities, questions and evidence

| Capability                                              | Question it earns                                                                    | Instrument treatment                                                                                                                                                                                        |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Z80 0.2, Z80 0.3 classic/NMOS, NMOS 6502, coherent 8086 | Does this behaviour depend on the instruction vocabulary or core revision?           | Native processor selector in experiment setup; machine shown beside the selected tape; disassembly uses that ISA. Changing machines starts a fresh condition.                                               |
| Ablation tables                                         | Which instruction families enable this behaviour?                                    | Keep inspectable opcode/family suppression with per-ISA validation. Explain that suppression changes subsequent decoding; never pretend vocabularies are interchangeable.                                   |
| Lethal zero and probability                             | Does making a zero instruction dangerous change the spatial outcome?                 | A labelled probability control; zero-density field plus culture comparison. The intervention concerns instruction starts, not every zero byte.                                                              |
| Fetch and start maps                                    | Did execution leave the focal tape, and which bytes were instruction starts?         | Population confinement observation when enabled; precise labelled fetch/start lanes in a selected culture encounter. Include discarded padding in the address range.                                        |
| Write counts                                            | Is output being actively written, even if its value stays the same?                  | A/B write totals in the encounter bench. Do not call a before/after difference a write count.                                                                                                               |
| Halt/trap reasons                                       | Why did this encounter stop?                                                         | Explicit budget/halt/trap readout next to steps and fetch lanes.                                                                                                                                            |
| Culture and self-copy tests                             | Does this candidate reproduce across fresh partners? Does it merely preserve itself? | Two-generation culture plus a distinct self-pair encounter. Retain sample size, seed, settings and machine manifest.                                                                                        |
| Marker planting/transfer                                | Does changing this byte change the offspring?                                        | Choose position/value, plant in the selected cell, show same-position value matches in the field; paired marked/unmarked cultures show retention against a control. Value matching is not ancestry.         |
| Partner inflow                                          | Does changing the initial partner change the resulting offspring?                    | Paired encounter assay: same parent/reset/seed, changed partner. Report changed output positions and whether the parent changed. This is a finite sensitivity test, not a complete causal-dependence proof. |
| Open/confined                                           | Where did instruction fetching leave A?                                              | Unknown/no observed departure/departure, last initiating encounter. Population colours do not certify a permanent species property.                                                                         |
| Per-ISA disassembly                                     | What could these bytes execute?                                                      | Static disassembly beside measured fetch/start lanes; explain self-modification and ablation can produce another path.                                                                                      |
| Checkpoints/manifests                                   | Can I keep the condition and evidence?                                               | Keep local `.acell` import/export; export assay JSON with full settings and manifest. Image/history exports record view and update. No promise of identical future scheduling.                              |
| Experiment presets                                      | What can I try first, and what should I watch?                                       | Plain-language names grouped into random beginnings, interventions and planted demonstrations; short question and look-for text. Provenance remains a secondary disclosure.                                 |

The live `GPUEngine` is the authoritative GPU population. The old serial
`LabEngine` is a reference-test service, not a replacement live backend. Rich
encounter records come from the separate, on-demand Zilion culture machine.

## 3. The observation instrument

### Field modes

All modes preserve the same cell positions, zoom, pan and selected address.
Switching a display must not reseed or advance the world.

| Mode            | Question                                          | Encoding and limits                                                                                                                                                                                 |
| --------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bytes           | What is physically present?                       | Existing per-byte palette and averaging. Byte value is not instruction role.                                                                                                                        |
| Tape identity   | Where are similar-looking repeated tapes?         | Stable complete-tape hash colours. Explicitly a visual fingerprint: colours/hashes can collide. Exact region counts verify identity. Do not label this biological species.                          |
| Open / confined | Where has fetching left A?                        | Grey unknown, blue no observed departure, amber departure into partner or padding. Optional instrumentation; changed tapes invalidate their observation.                                            |
| Zero density    | Where is zero-rich material accumulating?         | Fixed 0–100% scale of inherited bytes per tape. Do not label all zero-rich material inert or dead.                                                                                                  |
| Local diversity | Where are uniform patches and mixed boundaries?   | Exact distinct tapes among valid cells in a 3×3 lattice window divided by window size. This is local composition, not fitness or complexity; specify rectangular lattice coordinates on hex worlds. |
| Marker matches  | Where is the chosen value at the chosen position? | Neutral background and warm matching cells. Position and hex value always visible; spontaneous matches are possible.                                                                                |

Fetch heat, instruction starts and write activity belong in the encounter bench
in this release. The live soup deliberately omits detailed lane records for
speed; inventing live heat from a single culture would be misleading. Offer
fetch/start address strips and A/B write bars there, with their sample scope
visible. Genuine live heat and causal ancestry require a separately benchmarked
opt-in recorder and are not represented by fake data.

### Tools

- **Navigate:** wheel/pinch zoom and drag pan remain; add explicit zoom in/out
  and fit controls so they are discoverable and keyboard accessible.
- **Probe:** select a cell without losing the surrounding field. A compact
  floating readout gives address, bytes, zero fraction and refresh state. Arrow
  keys move the selected address. Refresh only its tape, at most twice a second
  while visible. Open the detailed encounter bench deliberately.
- **Region:** drag two corners, or choose endpoints numerically for keyboard
  and touch use. Outline the actual region. Sample while paused and show cell
  count, exact unique tapes, top share, zero fraction and marker-match fraction.
  The sample has an update stamp and never silently follows a moving population.
- **History:** opt-in sampled visual history, at most one frame per second and
  a bounded ring of 24 frames. A scrubber and replay navigate observations, not
  simulation state. Pauses while browsing; returning live resumes the current
  view without rewinding the soup. Frame update labels prevent ambiguity.
- **Compare:** pin a reference image and reveal it alongside the current view.
  Reference update and capture mode are visible; this is a visual comparison,
  not a claim that two worlds ran with equal conditions.
- **Record:** PNG snapshot and a bounded local timelapse export of recorded
  frames through browser-native media APIs where supported. Checkpoint remains
  the way to save the actual population. Assay JSON retains evidence.

These tools answer distinct questions. Cut a general drawing layer, arbitrary
annotations, a dashboard of gauges, independent camera controls in both halves,
and continuous full-population CPU analysis. They add complexity without helping
the first loop of observation and testing.

### Linked measurements

The lower-left original information instrument remains the fast byte/type/chart
readout. The observation tool expands a contextual dock only when needed. Region
statistics refer to the outlined region; the probe and bench refer to the marked
cell; history and reference show their update stamps. Numeric summaries accompany
coloured bars. Population composition and isolated test results must never share
an unlabeled axis. No result may be called heredity solely because it copies.

## 4. Experiment flow and language

1. **Set up:** choose an experiment and read what it makes visible. Processor,
   geometry, reset, mutation, chemistry and ablation form its condition. Changes
   that restart the population are clearly separated from view tools.
2. **Run:** the familiar play/speed controls remain; show paused/running and
   updates. Continuous local GPU work ends by 55 seconds and on hidden tabs.
3. **Observe:** select a mode, read its legend, zoom into a boundary, optionally
   record a short history and pin a reference.
4. **Probe:** select a cell or region, see an update-stamped sample, and open
   the detailed tape when it deserves testing.
5. **Test:** culture against fresh partners, compare a marked parent with the
   original, inspect a self-pair or test partner sensitivity. Each result names
   its actual procedure and scope.
6. **Record:** save checkpoint, image, sampled timelapse or assay evidence.

Rename opaque starts to descriptions such as “Random code · benign zero”,
“Random code · corrected Z80”, “Make zero lethal”, “Short-prefix copier”,
“Full-tape copier”, “Load and push”, and “Return-driven copier”. Keep stable
internal preset IDs for compatibility. Source cohort numbers belong in secondary
provenance text. The forbidden owner wording must not appear in labels, body
copy, accessible labels, tooltips, source repository names displayed in the UI,
or runtime error strings.

About: what Algocell is; how local execution works; copying versus inherited
differences; implementation and source credits. Guide: first experiment;
reading each field mode; probing/regions; encounter tests; history/comparison;
recording; limits and glossary. Preserve the existing illustrated explanations,
but retire stale descriptions of a separate legacy interface.

## 5. Layout and component specification

Two alternatives were considered. A permanent left experiment rail and right
inspector would make every feature visible but consume the landscape and repeat
the rejected generic workspace. The chosen composition extends the existing
edge instruments and uses transient, linked tools.

Desktop, 1440×900:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ [Algocell  Bytes ▾] [Navigate Probe Region History]    [▶ 8× ↺ save open … ?]│
│ [legend · scope · selected mode]                                           │
│                                                                            │
│                   full-size continuous lattice                            │
│                           ┌─ selected region ─┐                            │
│                           │                   │        [probe / bench]     │
│                           └───────────────────┘        [cell & byte tape]   │
│                                                        [test / record]     │
│ [original compact measurements] [contextual history / region / comparison] │
└────────────────────────────────────────────────────────────────────────────┘
```

Mobile, 390×844:

```text
┌──────────────────────────────┐
│ Algocell [Bytes ▾] [tools ▾] │
│ legend, scope                │
│                              │
│       pinchable lattice      │
│                              │
│                              │
│ [context tool: bounded sheet]│
│ [brief measurements]         │
│ [▶ 8× ↺ save open setup ?]   │
└──────────────────────────────┘
```

Observation controls use the same dark rounded shell and copper selection as
the existing toolbar. Native `<select>` controls keep platform keyboard and
popup behaviour. Stable DOM nodes survive counter updates. Disclosure heights
are bounded; tool panels scroll internally on phones, never horizontally expand
the document. Main mobile controls target 40–44px. No animated transitions of
the field or constantly changing panel widths.

Implementation seams: `ObservationTools.svelte` for tool state, history and
linked summaries; `LegacyInstrument.svelte` remains the run/setup shell and
passes the actual engine/canvas plus inspection callbacks. Engine owns GPU
field modes, small readbacks and marker writes. A plain TypeScript observation
module owns exact region summaries. `gpu/assays.ts` owns on-demand paired tests;
the bench consumes its records. Avoid moving thousands of unrelated legacy
lines just to satisfy a component count.

Accessibility: visible names for icons; input-modality-aware 1px focus rings;
Escape closes the active tool; no single-key shortcuts while typing; keyboard
probe and numeric region endpoints; legends and numeric values instead of
colour alone; screen-reader status only for explicit actions, not every frame;
reduced-motion disables automatic replay. Unsupported WebGPU retains help.

## 6. Performance and verification budget

The default simulator remains GPU-resident. No CPU pairing, ISA execution,
per-frame population readback or ancestry bookkeeping is added. The byte render
path and default simulation shader stay unchanged. Mode-specific cell metrics
are computed on GPU only when required, then sampled cheaply by the renderer.
No asynchronous work may overlap a reused staging buffer. Cell probes read one
word-padded tape; region summaries read once on request, with the world paused.
History is opt-in, bounded and visual; disabled history allocates/copies nothing.
Do not retain an unbounded collection of image bitmaps or media URLs.

Benchmark current HEAD and final with identical 200×200 L16/P32, 8192 proposed
pairs, 128-step budget, seed 6, k=4, benign zero, zero suppression, all five core
selections. Three short repetitions, warmup then reseed; 5-second bursts with
2-second idle gaps, no burst above 60 seconds. Report both compute-only and
rendered default behaviour and optional mode cost. A greater than 10% median
default regression requires investigation; do not call noisy measurements a
guarantee. Proposed instruction slots are not executed instructions.

Run check, lint, build and the real WebGPU Playwright suite; add meaningful
coverage for mode stability, region math, marker controls/tests, capture/replay,
comparison, exports, phone overflow, accessible controls and forbidden UI copy.
Capture every shipped mode and tool at 1440×900 and 390×844 under
`docs/review/evolution/`. Inspect images, record critique and iterate. No new
runtime dependencies; no pushing. Commits carry the requested co-author trailer.

## 7. Demanding critique and revisions

**Product critique:** “An extra row of buttons is another bolt-on.” Revision:
the view selector, legend, selected address and contextual tools become one
observation instrument with a single source of spatial context. Tests open from
the selected tape. Export acts on the thing being observed. Mode names describe
questions rather than implementation. Preserve the field's visual dominance.

**Product critique:** “All tools open at once will bury the phone.” Revision:
one contextual tool panel at a time; compact mobile controls; a bounded sheet;
comparison and history clearly exit back to live. Keep ordinary navigation as
the zero-panel default. Prefer a concise legend over an always-open essay.

**Research critique:** “Species colours, confinement and a bright marker can
suggest lineage that was never measured.” Revision: tape identity is explicitly
a fingerprint; exact region counting verifies sequences; confinement is an
encounter observation; marker mode is named matches. True descendants remain
deferred until ordered writes and a causal provenance model exist.

**Research critique:** “An inflow number can be falsely advertised as causal
independence.” Revision: run paired changed-partner encounters with the same
reset/seed and expose both outputs. Call it partner sensitivity and state sample
size. Marker retention is compared against the unmarked control. Self-pair
preservation is separate from culture reproduction.

**Engineering critique:** “Rich modes can quietly ruin the recovered speed.”
Revision: optional GPU cell reduction, small probes, paused region readback,
bounded opt-in visual history, and full execution records only in separate
encounters. Baseline evidence precedes changes; final evidence uses the same
runner. Default view stays on the original lean path.

This revised specification is the build target. The final review must distinguish
shipped tools from deferred instrumentation and state any remaining limitations.
