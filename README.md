# Algocell

A browser instrument for machine-code primordial soups. Watch neighbouring tapes meet, inspect what they execute, and test whether the material they produce can reproduce. Built with Svelte 5, TypeScript, WebGPU and [Zilion](https://github.com/NeoVand/zilion).

The lattice sits in one continuous workspace: a sidebar for World, Measure, Experiments, Guide and About; a thin rail for direct tools; and persistent run controls. Six observation modes, live cell probing, exact region measurements, visual history, reference comparison and the encounter bench share that shell. The population stays on the GPU. Available processors are Z80 compatibility 0.2, corrected Z80 0.3 classic and Zilog NMOS, NMOS 6502, and coherent-fetch 8086.

## Run locally

```sh
cd /Users/neo/repos/algocell-wt-app
npm ci
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

Open `http://127.0.0.1:5174/` in a WebGPU-capable browser. The instrument starts running. Tabs, tools, probes and views preserve playback and the current population. Work suspends while the document is hidden and resumes with the same playback choice. Simulation, assays, recordings and checkpoints remain in the browser.

For a production build:

```sh
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

## What to try

- Choose **Full-tape copier** in Experiments, review its staged condition, then choose **Start new world with these settings**. Compare Bytes, Tape identity and Open / confined from the header.
- Probe a cell, read its ISA-aware disassembly and run a two-transfer culture against fresh partners.
- Inspect fetched positions, instruction starts, A/B write counts and the stopping reason. Static disassembly is not a trace of self-modifying execution.
- Compare marker retention in **Short-prefix copier** and **Full-tape copier**. Use matched original and changed parents; field marker matches alone are not ancestry.
- Select a region to count exact tape identities and zero fraction at a recorded update.
- Record a bounded visual history, scrub or replay recent frames, pin a reference, and export PNG or WebM observations.
- Open **Export & checkpoints** for images, video and `.acell` population checkpoints. Assay JSON stays beside its result. Import previews a checkpoint before an explicit restore.

World separates live interventions from the staged next world. Same-length core swaps retain the tape bytes; seed and geometry edits take effect only through the explicit start action. Status items jump to their settings. Space plays/pauses, `.` steps, V/P/R/M/C select tools, 1–5 select sidebar tabs, and `?` opens the shortcut guide. All features live at `/`; there is no separate legacy interface.

## Measurement boundaries

Tape identity colours are hash fingerprints and may collide; region sequence counts and local diversity compare exact tapes. Open/confined records fetching outside the focal tape, including discarded padding, in the last measured initiating encounter. It does not measure partner-data reads, information independence or heredity.

Partner sensitivity changes one original partner byte while holding the parent and reset/chemistry seed fixed. An untouched byte can persist without being read. Marker transfer counts same-position value matches against an unmarked control; values may be regenerated or reacquired. These are finite tests, not causal ancestry or a branching inheritance protocol.

History records sampled images, not rewindable simulation state. A reference is a frozen view, not an independent control world. Checkpoints preserve population and settings but do not guarantee identical future GPU scheduling. BFF and hardware 8088 prefetch are outside the available processor contracts.

## Verification

```sh
npm run check
npm run lint
npm run build
npm run test:e2e
```

Playwright uses port 4183, one worker and installed Google Chrome with real WebGPU. Tests cover the 22 experiment presets, processor contracts, scientific fixtures, field measurements, interaction tools, exports, continuity, staged changes, stable controls, three desktop sizes, mobile layout and readable documentation without WebGPU. Local GPU workloads use short bursts and idle pauses.

The current design is in [docs/UNIFIED_DESIGN.md](docs/UNIFIED_DESIGN.md), with [task measurements](docs/review/unified/TASKS.md), [design review rounds](docs/review/unified/ROUNDS.md), [performance evidence](docs/review/unified/PERFORMANCE.md) and before/after screenshots in [docs/review/unified](docs/review/unified). The preceding [design evolution](docs/DESIGN_EVOLUTION.md) remains documented. The baseline runner can interleave the current engine with a frozen checkout:

```sh
BASELINE_REPO=/tmp/algocell-unified-6484a24 node scripts/evolution-benchmark.mjs
```

The public site deploys from `main`. This work stays on `feat/app-unified` and must not be pushed to `main`.

## Credits

Algocell and Zilion are by Mostafa “Neo” Mohsenvand. The work builds on the primordial-soup experiments of Agüera y Arcas and colleagues, Alexander Mordvintsev's implementation, and the Z80 work of Cicala and colleagues. Processor validation draws on SingleStepTests and Zilion's analytic and regression tests. See About and the experiment provenance notes for context.

MIT licensed.
