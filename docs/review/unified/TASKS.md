# Unified application task study

The paths are executable in `scripts/unified-tasks.mjs`. Baseline source is frozen at `6484a24`; captures and raw measurements live in `before/metrics.json`. The viewport for task comparison is 1440 × 900. The state atlas also covers 1280 × 800 and 1920 × 1080. Task success is defined by an observable result, not just opening a panel.

## People and intent

- **First-time visitor:** arrives from the README, wants to understand the evolving lattice, discover whether it is running, and make one meaningful observation without learning machine configuration.
- **Researcher:** reproduces a published preset, checks its world contract, adjusts controlled variables, measures a region, and saves evidence or a resumable checkpoint.
- **Power user:** follows one tape and its neighbors, examines an encounter, plants a marker, and compares time samples while the population continues evolving.
- **Teacher:** controls a live demonstration, switches views and scale frequently, explains zero chemistry, and revisits a recorded frame without losing the running world.

## Twelve repeatable tasks

| ID  | Task and success criterion                                                                   | People              | Expected frequency                    |
| --- | -------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------- |
| 01  | Open Guide, return to the lattice, pause and resume with a clearly identifiable run control. | Visitor             | Once on arrival; occasional reference |
| 02  | Compare identity, confinement and bytes; zoom and fit the lattice.                           | Visitor, teacher    | 10–30/session                         |
| 03  | Switch from legacy to corrected Z80 at the same tape length.                                 | Researcher          | 1–3/study                             |
| 04  | Load the short-prefix copier experiment and see its chosen world contract.                   | Researcher          | 1–6/session                           |
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

Each task starts from a fresh page at 1440 × 900, with the run playing. For the encounter and marker tasks, the short-prefix copier preset is installed during uncounted setup, ensuring meaningful, reproducible byte evidence. All other tasks start from defaults. Setup and screenshot capture are excluded from task cost. Browser and server are closed after the run, and each task has a GPU rest interval.

A click is one actual pointer activation of a control. Number and range edits begin with one click followed by typing; a native selector uses one click followed by keyboard option selection. Thus these figures compare a consistent mixed pointer-and-keyboard path, not an invented count of native menu internals. Escape and other keyboard commands are recorded in paths but do not add pointer clicks. Pointer travel is the sum of straight-line distances between the centers of clicked targets, starting at (720, 450); scrolling is not converted into artificial pointer travel. Results round only the final pixel sum.

A surface is an independently positioned interaction container visited by the pointer: the floating widget, speed popover, observation strip, observation dock, settings panel, encounter dialog or help dialog. In the unified version, the corresponding surface names are the header/run controls, rail, sidebar and status bar. Sections inside the same sidebar count as one surface. The lattice canvas counts only when directly manipulated.

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
