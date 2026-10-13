# Algocell

A browser laboratory for machine-code primordial soups. Watch neighboring tapes meet, inspect what they execute, and test whether the bytes they produce can reproduce. Built with Svelte 5, TypeScript, WebGPU and [Zilion](https://github.com/NeoVand/zilion).

The research workspace supports historical Z80 compatibility 0.2, corrected Z80 0.3, the Zilog NMOS variant, NMOS 6502 and coherent-fetch 8086. Paper presets make the processor, world, reset, budget and chemistry explicit. A live browser run is a new experiment, not a replay of an archived paper trajectory.

## Run locally

From this worktree:

```sh
cd /Users/neo/repos/algocell-wt-app
npm ci
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

Open `http://127.0.0.1:5174/` in a WebGPU-capable browser. The workspace starts paused. Run uses bursts of at most 60 seconds; Step advances one population update. Simulation, assays, imports and exports stay in the browser. No account or external runtime service is required.

To inspect a production build:

```sh
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

## What to try

- Select a paper preset and read its source, declared conditions and limitations.
- Compare zero-byte fraction, the most abundant exact tape's share, and distinct-tape diversity. These describe composition; none alone detects reproduction or heredity.
- Select a cell to see ISA-aware disassembly. A static reading is not a trace of self-modifying execution.
- Run a fresh-partner culture test. Inspect copying, continued reproduction and instruction-stream partner entry separately.
- Compare a short-displacement copier with a full-tape copier. Confined fetching can accompany very different fates for a payload marker.
- Export the complete condition and its Zilion manifest when keeping a result.

The **Legacy instrument** at `/legacy` preserves the original optimized Z80 lattice, square/hex rendering, visual adjustments, stored presets and `.acell` snapshots. Its frozen core is historical compatibility 0.2. Its old population scheduler is GPU-order dependent; legacy snapshots do not contain enough continuation state to guarantee exact replay.

## Scientific boundaries

The app separates the processor from its world and the population scheduler. Zilion's SHA-256 manifest identifies a machine; population settings, random streams and mutations need their own record. The new workspace uses a versioned deterministic host scheduler and GPU encounter execution, so its throughput and trajectories differ from the legacy all-GPU implementation.

“Confined” refers to instruction-stream fetching within the initiating tape. It does not mean no partner data was read, no external influence, or inherited variation. Culture results are finite, condition-specific tests. A marker's matching value is not proof of its causal ancestry. Seeded mechanism examples are constructions, not spontaneous discoveries. The paper's registered multi-world, branching and serial-transfer studies remain distinct from these short browser demonstrations.

BFF is not offered because it is excluded from the Zilion 0.3.1 npm exports.

## Development and verification

```sh
npm run check
npm run lint
npm run build
npm run test:e2e
```

Playwright uses a dedicated local Vite server on port 4183, one worker and installed Google Chrome (`channel: 'chrome'`) with real WebGPU. Install Chrome if it is unavailable. Tests exercise the workspace, presets, ISAs, inspector, documentation, core scientific fixtures, checkpoint continuation and import recovery. GPU tests require an available WebGPU adapter and fail explicitly if it is unavailable; a separate test checks the readable preview with WebGPU disabled.

The implementation plan, audit, source catalogue, design system and remaining roadmap are in [docs/APP_PLAN.md](docs/APP_PLAN.md). Detailed studies: [app](docs/study-app.md), [engine](docs/study-engine.md), [paper](docs/study-paper.md). Review screenshots live in [docs/screenshots](docs/screenshots).

The public GitHub Pages site deploys from `main`. This development branch must not be pushed to `main` without the owner's separate deployment decision.

## Credits

Algocell and Zilion are by Mostafa “Neo” Mohsenvand. The work builds on the primordial-soup experiments of Agüera y Arcas and colleagues, Alexander Mordvintsev's implementation, and the Z80 work of Cicala and colleagues. Processor validation draws on the SingleStepTests suites and Zilion's analytic and regression tests. See the app's About page and source-bound preset notes for scientific context.

MIT licensed.
