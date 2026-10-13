# Algocell evolution review

Branch: `feat/app-legacy-base`. Starting revision: `ee67698dd7b743e838eacd684855a3d9fcefb362`.
The design was written and committed as `1ba8712` before application code changed.
No push or change to `main` was made. No runtime dependency was added.

## What evolved

The full-screen square/hex byte lattice, camera interaction, original palettes,
copper actions, charcoal surfaces and compact transport remain the instrument's
base. A visible observation strip connects six GPU field views to a persistent
probe, exact sampled region measurements, bounded visual history, reference
comparison and local recordings. The encounter bench sits beside the field and
connects the selected tape to static disassembly and measured execution.

Population views:

- Bytes retains the original renderer.
- Tape identity uses stable complete-tape hash fingerprints; colours can collide.
- Open / confined distinguishes unknown, no observed departure and fetching
  outside A, including padding. A changed tape loses its observation.
- Zero density measures zero bytes divided by inherited tape length.
- Local diversity compares exact tapes in the valid 3×3 coordinate window.
- Marker matches shows a value at one position, without claiming ancestry.

The population stays on the GPU. Optional field reductions run once per changed
population before rendering and are cached while paused. The default soup shader
omits detailed encounter records. A probe reads at most one 128-byte tape through
a serialized staging buffer; the original hover now uses this path too. Exact
region counting occurs once on demand while paused. History retains at most 24
small view images at one sample per second, only when enabled.

The separate encounter machine supplies fetched addresses, instruction starts,
A/B write counts, registers, halt and trap reasons. The bench offers two-transfer
culture, an identical-partner self-pair, matched one-byte partner perturbations,
and matched marked/original-parent serial transfers. All offspring enter the next
transfer. Evidence export keeps records, seeds, settings and the Zilion manifest.

Experiments now have plain-language names. About and Guide explain observations,
methods and limits in the instrument's own language. Existing processor choices,
ablation tables, chemistry controls, checkpoint import/export and user presets
remain available. Browser zoom is enabled; stale Z80-only share metadata and the
unsupported throughput claim were removed.

## Scientific validation

New real-GPU fixtures verify that a short-prefix copier can continue producing
its pattern while erasing the introduced byte, whereas a full-tape copier keeps
it through both transfers. A halted 6502 or 8086 preserves the deliberately
changed partner byte without fetching it: partner sensitivity reports the output
difference and does not infer that it was read. Self-pair results separately show
preservation and actual write counts.

GPU field fixtures verify exact local diversity, word-padding exclusion,
zero fractions, same-position marker matching, one-byte planting, narrow probe
reads and region counts. Mode changes preserve population bytes, camera and
update counter. All modes compile and render on square and 19-byte hex worlds;
optional confinement executes across all five core selections.

## Review corrections

Independent static review identified and corrected observation semantics rather
than merely changing appearance:

- Current-tape confinement is invalidated when an encounter changes the tape.
- Rapid paused probe selection must resolve the latest requested cell.
- Changing region coordinates cannot leave a result visually attributed to the
  new rectangle before another sample is taken.
- A marker remains consistent with its legend after changing tape length.
- Discarded padding has no retained final value; missing output is never filled
  with an invented zero.
- Observation metadata comes from applied engine settings, not unapplied form
  drafts. Configuration and observation operations are coordinated.
- Closing replay or changing views cannot restart a hidden history overlay.

## Limits retained deliberately

There is no causal byte-ancestry tracker or descendant map. Marker fields and
transfer results measure value matches, which can be regenerated or reacquired.
Partner perturbation is a finite sensitivity assay, not a complete inflow or
information-independence proof. Full branching inheritance studies remain outside
these short local tests.

Live full-population fetch heat, instruction-start heat and write heat are not
fabricated from sampled encounters; those lenses appear in the encounter bench.
Adding detailed records to every live lane requires a separately measured
opt-in recorder. Identity colours remain fingerprints; exact region and local
diversity measurements provide collision-free composition checks.

History is visual and sampled, not reversible simulation state. Reference
comparison holds an image, not a second simulated control. WebM export depends
on browser MediaRecorder support. Checkpoints keep bytes and settings; restoring
restarts counters and random continuation. They do not contain a Zilion manifest
or guarantee an identical future trajectory. Assay JSON includes the manifest.
Image and video exports identify their mode and update but have no manifest
sidecar. Region trends have no CSV/JSON export. Existing population charts group
by hashes; region and local-diversity measurements supply exact comparisons.

Ablation uses validated per-ISA patterns and families, with match counts and
chips; a full browsable opcode matrix remains future work. BFF and hardware 8088
prefetch remain unavailable.

## Run and reproduce

```sh
cd /Users/neo/repos/algocell-wt-app
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

Open `http://127.0.0.1:5174/` in a WebGPU-capable browser. For a fresh checkout,
run `npm ci` first. The app pauses runs after 55 seconds and when hidden.
The test and capture scripts use short GPU bursts with idle pauses.

The baseline procedure, environment and original screenshots are recorded in
[baseline.md](baseline.md).

## Performance evidence

Measurements use the same M4 MacBook Air, Chrome/Metal WebGPU, 200×200 square
world, L16/P32, seed 6, 8,192 proposed pairs, 128-step budget, mutation exponent
4, benign zero and no ablation. Each core keeps its original reset and chemistry
contract. Rates count completed updates; proposed instruction slots are not
executed instructions. GPU activity is broken into short bursts with idle gaps.

The full-application comparison runs both revisions in matching Vite servers,
with the native 8× animation loop, 1440×900 canvas, Bytes/Navigate selected and
the usual collapsed statistics panel. Both applications are preloaded; each run
warms up, resets, drains the GPU, then measures five seconds. Three paired
repeats alternate build order, with pages closed during three-second pauses.
No builds, checks or other GPU jobs run concurrently.

| Core                   | Baseline updates/s | Evolution updates/s | Evolution fps | Median paired change |
| ---------------------- | -----------------: | ------------------: | ------------: | -------------------: |
| Z80 compat 0.2         |              479.2 |               479.6 |          60.0 |               +0.11% |
| Z80 compat 0.3 classic |              479.9 |               479.3 |          59.9 |               −0.08% |
| Z80 compat 0.3 NMOS    |              480.1 |               479.1 |          59.9 |               −0.19% |
| MOS 6502               |              479.4 |               479.9 |          60.0 |               +0.10% |
| Coherent 8086          |              458.1 |               438.4 |          54.8 |               −0.68% |

Independent medians and median paired ratios answer different questions. For
8086 the independent-median change is −4.3%; its paired change is −0.68%.
The default view adds no soup, probe or hash readback. These frame-limited
results do not establish unchanged maximum compute headroom.
Raw data and exact settings: [app-benchmark.json](app-benchmark.json).

An earlier unconstrained compute/render sweep is retained in
[benchmark-exploratory.json](benchmark-exploratory.json). It had broad ranges,
and a build overlapped early samples; it is not a clean performance gate.
A quiet confirmation used exactly 2,048 completed updates, five alternating
paired repeats, and the original one-update warmup:

| Core           | Baseline updates/s | Evolution updates/s | Median paired change |
| -------------- | -----------------: | ------------------: | -------------------: |
| Z80 compat 0.2 |              756.2 |               763.0 |               +0.42% |
| MOS 6502       |            1,859.6 |             1,732.9 |               −5.15% |
| Coherent 8086  |              449.6 |               434.8 |               +0.51% |

The 6502 decrease warrants investigation even though its interface stays at
60 fps. Equal updates do not imply identical executed instructions in evolving,
parallel worlds. A one-update warmup also leaves setup/compilation as a possible
confound. The default compute loops, bindings, parameter writes and buffer sizes
are unchanged. All five uninstrumented WGSL shaders have identical SHA-256
hashes after removing comments and whitespace; this structural evidence does
not by itself disprove a measured slowdown. See
[benchmark-confirmatory.json](benchmark-confirmatory.json) and
[shader-contract.json](shader-contract.json).

A final 6502 control investigation uses 512 warmup updates, a completed reset
upload, and five paired 2,048-update runs with the same alternating order and
three-second pauses. The same-version control uses byte-identical engine bundles
(their SHA-256 hashes are recorded in the raw data):

| Warmed comparison             | First build updates/s | Second build updates/s | Median paired change | Individual paired range |
| ----------------------------- | --------------------: | ---------------------: | -------------------: | ----------------------: |
| Baseline / identical baseline |               2,017.1 |                2,009.4 |               +0.56% |      −10.67% to +22.37% |
| Baseline / evolution          |               1,932.1 |                2,139.8 |              +11.98% |       −1.32% to +90.04% |

The earlier 6502 decrease did not reproduce. The identical-build control exposes
enough device/workload/harness variability that the warmed comparison must not
be advertised as a speedup. No reproducible default slowdown was established;
the most stable observation is the unchanged practical full-app frame rate and
throughput. A small change in unconstrained headroom cannot be excluded by these
short local samples. All series are retained, including unfavourable results:
[benchmark-6502-aa.json](benchmark-6502-aa.json) and
[benchmark-6502-warm.json](benchmark-6502-warm.json).

The full-app series used `70184f1`; subsequent changes corrected legends,
tooltips, explicit framing actions and mobile panel CSS. They did not change the
default simulation or add default readbacks. The final warmed engine series used
`7227f8f`. Source revisions and exact bundle hashes are embedded in the artifacts.

The six-view cost smoke uses three 512-update rendered runs per view. Median
updates/s were Bytes 528.2, identity 522.4, confinement 524.5, zeros 521.9,
diversity 525.0 and marker 529.1. Wide run ranges prevent precise mode rankings;
the optional views produced no obvious throughput cliff in this sample.
See [mode-costs.json](mode-costs.json).

## Visual critique and iteration

The observation strip follows the original compact charcoal/copper controls;
the field remains the largest surface. Desktop measurement and encounter panels
sit beside it. Native selectors keep their DOM identity during updates, and
focus outlines follow keyboard use.

The initial phone region panel covered too much of the field. It was shortened
to 49dvh, its values arranged in two columns, and redundant decoration removed.
Single region samples now show visible chart dots. Opening the encounter bench
hides the underlying probe panel instead of leaving a translucent ghost.

Final screenshot inspection found that approximate CSS legend colours did not
match the GPU's exact endpoints, and that a remaining padding tooltip invented
a retained zero. The final UI uses shader-matched scales and “not retained” for
discarded padding. Phone Fit views expose the aspect ratio of a wide world;
zoomed captures demonstrate actual navigation rather than cropping the image
afterward. Explicit selection-framing controls make this reframing available
within the probe and region workflows.

Expanded marker controls also needed the same height budget as Region. All
mobile observation panels now share the 49dvh cap and a sticky opaque header.
The marked cell remains visible above the panel; its framing and Close controls
remain accessible while scrolling. The header reaches the inner border instead
of exposing a strip of scrolled content above it.

The final collection contains 58 screenshots: 29 views/tools at both 1440×900
and 390×844, plus PNG and WebM exports at both sizes. See
[SCREENSHOTS.md](SCREENSHOTS.md) for direct links to every capture, including
explicit Fit, selection framing, all encounter lenses and expanded disassembly.
The capture report records no page/capture errors or horizontal overflow.

All 58 captures were visually inspected across the design and scientific review.
The final review accepted the field modes, observation tools and encounter
results at both sizes. Mobile bench screenshots deliberately scroll to the
measurement being demonstrated; their internal scroll does not hide controls
permanently or force the page to overflow.

## Validation

- `npm run check`: 537 files, zero errors and zero warnings on the final source.
- `npm run lint`: Prettier and ESLint pass, including the evidence scripts.
- `npm run build`: production build passes on the final source.
- `npm run test:e2e`: all 59 tests passed at `02eb485`, covering the 22 presets,
  real-GPU scientific contracts, observation tools, exports, checkpoints,
  no-WebGPU help, stable native controls and keyboard focus.
- After the mobile height and sticky-header changes at `7227f8f`, all 14 tests
  in `tests/evolution-tools.e2e.ts` passed. After the final header alignment
  correction at `59cd038`, both affected desktop/mobile probe-and-marker tests
  passed again, including panel height, unobscured selection, reachable framing
  and direct sticky-header alignment assertions.
- The final capture report contains 58 successful screenshots, with no missing
  files, page/capture errors or horizontal overflow. PNG and WebM exports were
  produced at both viewport sizes.

The default simulation shader contract matches the baseline for all five core
selections. The new Svelte components also pass the Svelte MCP analysis with no
issues. All local GPU work used bounded bursts with pauses; no browser or GPU
job remains running after the review.

## Reproduce measurements

Create the frozen comparison checkout if it does not already exist:

```sh
git worktree add --detach /tmp/algocell-evolution-ee67698 ee67698dd7b743e838eacd684855a3d9fcefb362
ln -s /Users/neo/repos/algocell-wt-app/node_modules /tmp/algocell-evolution-ee67698/node_modules
```

Run only one GPU job at a time, with no concurrent checks/builds. The runners
close their pages between short bursts and supply idle pauses.

```sh
# Full native application, all five core selections.
node scripts/evolution-app-benchmark.mjs

# Default shader contract comparison; no GPU work.
node scripts/evolution-shader-contract.mjs

# Fixed-work confirmation, original one-update warmup.
BASELINE_REPO=/tmp/algocell-evolution-ee67698 BENCH_OUTPUT=docs/review/evolution/benchmark-confirmatory.json CASES=z80:0.2:classic,m6502:0.3:nmos,i8086:0.3:nmos MODES=compute FIXED_UPDATES=2048 REPEATS=5 BENCH_ORDER=alternate PAUSE_MS=3000 node scripts/evolution-benchmark.mjs

# Six optional-view cost samples.
BENCH_OUTPUT=docs/review/evolution/mode-costs.json CASES=z80:0.2:classic MODES=rendered FIELD_MODES=bytes,identity,confinement,zeros,diversity,marker FIXED_UPDATES=512 REPEATS=3 PAUSE_MS=3000 node scripts/evolution-benchmark.mjs

# Warmed 6502 same-version control.
BASELINE_REPO=/tmp/algocell-evolution-ee67698 BENCH_REPO=/tmp/algocell-evolution-ee67698 BENCH_OUTPUT=docs/review/evolution/benchmark-6502-aa.json CASES=m6502:0.3:nmos MODES=compute FIXED_UPDATES=2048 REPEATS=5 BENCH_ORDER=alternate PAUSE_MS=3000 WARMUP_UPDATES=512 DRAIN_RESET=1 node scripts/evolution-benchmark.mjs

# Warmed 6502 baseline/current comparison.
BASELINE_REPO=/tmp/algocell-evolution-ee67698 BENCH_OUTPUT=docs/review/evolution/benchmark-6502-warm.json CASES=m6502:0.3:nmos MODES=compute FIXED_UPDATES=2048 REPEATS=5 BENCH_ORDER=alternate PAUSE_MS=3000 WARMUP_UPDATES=512 DRAIN_RESET=1 node scripts/evolution-benchmark.mjs

# Every final viewport/mode/tool, plus local PNG/WebM exports.
node scripts/evolution-screenshots.mjs
```
