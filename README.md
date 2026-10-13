# Algocell

A browser instrument for machine-code primordial soups. Watch neighbouring tapes meet, inspect what they execute, and test whether the material they produce can reproduce. Built with Svelte 5, TypeScript, WebGPU and [Zilion](https://github.com/NeoVand/zilion).

The original full-screen lattice now has six observation modes, live cell probing, exact region measurements, visual history, reference comparison and a linked encounter bench. The population stays on the GPU. Available processors are Z80 compatibility 0.2, corrected Z80 0.3 classic and Zilog NMOS, NMOS 6502, and coherent-fetch 8086.

## Run locally

```sh
cd /Users/neo/repos/algocell-wt-app
npm ci
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

Open `http://127.0.0.1:5174/` in a WebGPU-capable browser. The original instrument starts running. Play/Pause controls the population; runs pause after 55 seconds and when hidden. Simulation, assays, recordings and checkpoints remain in the browser.

For a production build:

```sh
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

## What to try

- Choose **Full-tape copier** in Settings → Experiments, then compare Bytes, Tape identity and Open / confined.
- Probe a cell, read its ISA-aware disassembly and run a two-transfer culture against fresh partners.
- Inspect fetched positions, instruction starts, A/B write counts and the stopping reason. Static disassembly is not a trace of self-modifying execution.
- Compare marker retention in **Short-prefix copier** and **Full-tape copier**. Use matched original and changed parents; field marker matches alone are not ancestry.
- Select a region to count exact tape identities and zero fraction at a recorded update.
- Record a bounded visual history, scrub or replay recent frames, pin a reference, and export PNG or WebM observations.
- Save `.acell` checkpoints for population bytes and JSON assay evidence for settings, records and machine manifests.

All features live at `/`; there is no separate legacy interface.

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

Playwright uses port 4183, one worker and installed Google Chrome with real WebGPU. Tests cover the 22 experiment presets, processor contracts, scientific fixtures, field measurements, interaction tools, exports, mobile layout and readable documentation without WebGPU. Local GPU workloads use short bursts and idle pauses.

The design and critique are in [docs/DESIGN_EVOLUTION.md](docs/DESIGN_EVOLUTION.md). Before/after screenshots, benchmark measurements and the final review are in [docs/review/evolution](docs/review/evolution). The baseline runner can interleave the current engine with a frozen checkout:

```sh
BASELINE_REPO=/tmp/algocell-evolution-ee67698 node scripts/evolution-benchmark.mjs
```

The public site deploys from `main`. This work stays on `feat/app-legacy-base` and must not be pushed to `main`.

## Credits

Algocell and Zilion are by Mostafa “Neo” Mohsenvand. The work builds on the primordial-soup experiments of Agüera y Arcas and colleagues, Alexander Mordvintsev's implementation, and the Z80 work of Cicala and colleagues. Processor validation draws on SingleStepTests and Zilion's analytic and regression tests. See About and the experiment provenance notes for context.

MIT licensed.
