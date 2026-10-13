# Unified application task study

The paths are executable in `scripts/unified-tasks.mjs`. Baseline source is frozen at `6484a24`; captures and raw measurements live in `before/metrics.json`. The viewport for task comparison is 1440 × 900. The state atlas also covers 1280 × 800 and 1920 × 1080. Task success is defined by an observable result, not just opening a panel.

## People and intent

- **First-time visitor:** arrives from the README, wants to understand the evolving lattice, discover whether it is running, and make one meaningful observation without learning machine configuration.
- **Researcher:** reproduces a published preset, checks its world contract, adjusts controlled variables, measures a region, and saves evidence or a resumable checkpoint.
- **Power user:** follows one tape and its neighbors, examines an encounter, plants a marker that identifies matching byte values, and compares time samples while the population continues evolving.
- **Teacher:** controls a live demonstration, switches views and scale frequently, explains zero chemistry, and revisits a recorded frame without losing the running world.

## Twelve repeatable tasks

| ID  | Task and success criterion                                                                   | People              | Expected frequency                    |
| --- | -------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------- |
| 01  | Open Guide, return to the lattice, pause and resume with a clearly identifiable run control. | Visitor             | Once on arrival; occasional reference |
| 02  | Compare identity, confinement and bytes; zoom and fit the lattice.                           | Visitor, teacher    | 10–30/session                         |
| 03  | Switch from legacy to corrected Z80 at the same tape length.                                 | Researcher          | 1–3/study                             |
| 04  | Load the full-tape copier experiment and see its chosen world contract.                      | Researcher          | 1–6/session                           |
| 05  | Set 2× speed, adjust mutation and zero chemistry during a demonstration.                     | Teacher             | 5–20/session                          |
| 06  | Probe cell 0, follow the next cell and center the selected tape.                             | Power user          | 20–100/session                        |
| 07  | Probe, open an encounter, run the self-pair comparison and inspect writes.                   | Power user          | 5–20/session                          |
| 08  | Measure a 2 × 2 region and frame the measured area.                                          | Researcher, teacher | 5–15/session                          |
| 09  | Plant a marker at byte 8 and reveal matching tapes.                                          | Power user          | 2–10/session                          |
| 10  | Record recent samples, scrub to an earlier sample, then return live.                         | Power user, teacher | 3–10/session                          |
| 11  | Pin a reference and move the comparison split.                                               | Researcher, teacher | 3–10/session                          |
| 12  | Download a PNG snapshot and a resumable checkpoint.                                          | Researcher, teacher | 1–5/session                           |

Frequency values are use-model estimates, not observations from a user study. Tasks 03 and 04 are infrequent world work; 02 and 06 dominate recurring inspection. Their placement should reflect that distinction.

## Measurement method

Each task starts from a fresh page at 1440 × 900, with the run playing. For the encounter and marker tasks, the full-tape copier preset is installed during uncounted setup, ensuring meaningful, reproducible byte evidence. All other tasks start from defaults. Setup and screenshot capture are excluded from task cost. Browser and server are closed after the run, and each task has a GPU rest interval.

A click is one actual pointer activation of a control. Number and range edits begin with one click followed by typing; a native selector uses one click followed by keyboard option selection. Thus these figures compare a consistent mixed pointer-and-keyboard path, not an invented count of native menu internals. Escape and other keyboard commands are recorded in paths but do not add pointer clicks. Pointer travel is the sum of straight-line distances between the centers of clicked targets, starting at (720, 450); scrolling is not converted into artificial pointer travel. Results round only the final pixel sum.

A surface is an independently positioned interaction container visited by the pointer: the floating widget, speed popover, observation strip, observation dock, settings panel, encounter dialog or help dialog. In the unified version, the corresponding surface names are the Look header, tool rail, sidebar and transport/status bar. Sections inside the same sidebar count as one surface. The lattice canvas counts only when directly manipulated.

A page-side observer records every playing-to-paused transition during the path. These include an intentional pause in task 01 and incidental stops elsewhere; the path makes the distinction reviewable. `engine.reset()` invocations count resets. Engine object replacement counts recreation. Internal `setMachine()` calls are configuration requests. Separate `createPipelines()` and `createBuffers()` hooks count actual pipeline rebuilds and population-buffer recreations, because configuration can recompile pipelines while preserving both population buffers and the JavaScript engine object. These are real runtime measurements, not deductions from screenshots.

The harness writes the full ordered action path, errors, state screenshots and metrics JSON. A failed path is recorded as a failure; it is never assigned guessed metrics. Counts are usability proxies, not a substitute for observed human task completion time or accessibility testing.

## Running the paths

```sh
UNIFIED_ROOT=/tmp/algocell-unified-baseline-6484a24 node scripts/unified-tasks.mjs
UNIFIED_ATLAS=none UNIFIED_TASKS=02-look-and-frame node scripts/unified-tasks.mjs
UNIFIED_PHASE=after UNIFIED_ROUND=1 node scripts/unified-tasks.mjs
```

The script owns port 4185 unless `UNIFIED_URL` points to an already-running server. It never touches the reviewer server on port 5175.

## Measured baseline

The final baseline run completed all twelve paths without errors. Pipeline rebuilds are distinct from engine or population-buffer recreation: the engine object survived every measured path. Loading the preset recreated population buffers once; switching the same-length core preserved those buffers but reseeded the population.

| Task                | Clicks | Pointer px | Surfaces | Pauses | Resets | Pipeline rebuilds |
| ------------------- | -----: | ---------: | -------: | -----: | -----: | ----------------: |
| 01-first-visit      |      4 |       2120 |        2 |      1 |      0 |                 0 |
| 02-look-and-frame   |      5 |       1002 |        1 |      1 |      0 |                 2 |
| 03-change-core      |      3 |       1168 |        2 |      1 |      1 |                 1 |
| 04-reproduce-preset |      3 |        981 |        2 |      1 |      1 |                 1 |
| 05-live-controls    |      6 |       1749 |        3 |      1 |      0 |                 2 |
| 06-follow-probe     |      5 |       2157 |        2 |      0 |      0 |                 0 |
| 07-encounter        |      7 |       3194 |        3 |      1 |      0 |                 0 |
| 08-region-measure   |      7 |       2818 |        2 |      1 |      0 |                 0 |
| 09-follow-marker    |      8 |       5016 |        2 |      1 |      0 |                 0 |
| 10-history          |      5 |       2160 |        2 |      1 |      0 |                 0 |
| 11-compare          |      4 |       2048 |        2 |      0 |      0 |                 0 |
| 12-export           |      3 |       2167 |        3 |      1 |      0 |                 0 |

Task 01 includes the one deliberate pause/resume. All other recorded pauses are incidental. The complete ordered paths and event timestamps are in `before/metrics.json`.

## Unified paths and review rounds

The after paths preserve all twelve baseline outcomes and use the persistent sidebar, header and tool rail. A custom listbox uses one pointer click, Home, ArrowDown and Enter, matching the baseline’s one-click plus keyboard selection method. Opening its choices is a real UI action and the harness verifies the resulting value. Task 04 adds the explicit staged-world confirmation; this extra action protects the population. It is counted rather than hidden in setup. Task 01’s requested pause and task 04’s requested reset are separated from incidental disruptions in after metrics.

The full-tape preset is `displacement-32`; earlier raw capture metadata called it short-prefix. Marker matches establish shared byte values, not ancestry. The task wording above corrects both terms. No before measurements were changed.

Each after round records all twelve task results at 1440 × 900. Its atlas includes every main sidebar tab, probe, region, history, encounter and collapsed-sidebar state at 1280 × 800, 1440 × 900 and 1920 × 1080. Detailed view, core, listbox, guide, export, replay and encounter-lens states are recorded at 1440 × 900. Screenshots and metrics are scoped to `after/round-N/`; review changes occur between runs. Each round runs against an immutable copy of the runtime source, with the revision, timestamp and source-content hash recorded in its metrics JSON. This keeps live development and hot reloads from contaminating a measured path.

The final round additionally records checkpoint import before explicit restoration and after restoration, the unavailable-WebGPU state and its working Guide, the full staged-world summary, the live core-change notice, Guide subsection memory across an About visit, and a 390 × 844 mobile regression atlas. The mobile paths activate World, Probe, Region and Encounter with real pointer clicks while the sidebar is expanded; the initial collapsed state shows the full lattice.

The baseline atlas has 114 successful screenshots in total: 55 original states and 59 additional discrete states in `before/additional-states.json`. The supplement includes every experiment and core, the four built-in personal presets, save/overwrite/delete, ablation, all explanatory hints, species inspection and its byte hover, recorded-frame replay, and the unavailable-WebGPU state with its accessible Guide. The original task JSON and numeric baseline remain unchanged.

<!-- UNIFIED_RESULTS_START -->

## Before / after task metrics — review round 4

Each entry is baseline → unified. Pointer travel is in pixels at 1440 × 900. Pauses and resets include the requested transport pause and requested preset restart, with incidental disruptions listed separately below.

| Task                |  Clicks |    Pointer px | Surfaces | Pauses | Resets |
| ------------------- | ------: | ------------: | -------: | -----: | -----: |
| 01-first-visit      |   4 → 3 |   2120 → 2281 |    2 → 2 |  1 → 1 |  0 → 0 |
| 02-look-and-frame   |   5 → 5 |    1002 → 903 |    1 → 1 |  1 → 0 |  0 → 0 |
| 03-change-core      |   3 → 1 |    1168 → 539 |    2 → 1 |  1 → 0 |  1 → 0 |
| 04-reproduce-preset |   3 → 3 |    981 → 1217 |    2 → 1 |  1 → 0 |  1 → 1 |
| 05-live-controls    |   6 → 3 |   1749 → 1958 |    3 → 2 |  1 → 0 |  0 → 0 |
| 06-follow-probe     |   5 → 5 |    2157 → 950 |    2 → 2 |  0 → 0 |  0 → 0 |
| 07-encounter        |   7 → 7 |   3194 → 1643 |    3 → 2 |  1 → 0 |  0 → 0 |
| 08-region-measure   |   7 → 7 |   2818 → 1414 |    2 → 2 |  1 → 0 |  0 → 0 |
| 09-follow-marker    |   8 → 8 |   5016 → 2117 |    2 → 2 |  1 → 0 |  0 → 0 |
| 10-history          |   5 → 6 |   2160 → 1763 |    2 → 1 |  1 → 0 |  0 → 0 |
| 11-compare          |   4 → 4 |    2048 → 983 |    2 → 2 |  0 → 0 |  0 → 0 |
| 12-export           |   3 → 3 |   2167 → 1464 |    3 → 1 |  1 → 0 |  0 → 0 |
| Total               | 60 → 55 | 26580 → 17232 |  26 → 19 | 10 → 1 |  2 → 1 |

Incidental pauses: 9 → 0. Incidental resets: 1 → 0. World confirmation remains a counted, intentional action in task 04.

<!-- UNIFIED_RESULTS_END -->

For the combined Export section introduced in review round 2, task 12 uses the permanent “Export & checkpoints” footer action, then PNG and checkpoint export. The alternate route through Measure → Export was also measured: four clicks and 1,397 px, versus three clicks and 1,482 px through the footer. Both actual paths are retained in the round 2 JSON; the primary task uses the shorter click path.
