# The neighbourhood rule and free-running cells

11 November 2026. A first look, not a result: one to five runs per setting.

## Why

The free-running world (`docs/free-running.md`) laid a program's byte order along the sheet, so order became a
direction in space: lines, rectangles, no round growth. The author proposed a rule of the convolution kind
instead: take the programs around a cell as one program, let it write only inside that cell, update every cell at
once. The order of bytes then lives inside the cell, as in the cell soups, and space only knows which cell is
next to which.

## The rule (`interaction: 'neighbourhood'`)

1. Every cell holds a 16-byte tape and has its own processor.
2. At each update the processor's memory is a ring of two halves: the tapes of `gather` of its four neighbours
   (1, 2 or 4), met from a random neighbour onwards in a random sense, and then the cell's own tape (repeated to
   the same length). Execution starts at the first byte of the first neighbour with every register at zero, so
   the stack grows down from the end of the cell's own tape.
3. The processor may read and write anywhere in that ring, but only what it writes to its own tape is kept.
   Writes to a neighbour's tape change a private copy that is thrown away.
4. A run ends when it has written every byte of the cell, or at HALT, or after `steps` instructions (see "How
   long to run").
5. All cells are updated at once from the old lattice. No two processors ever want the same byte, so nothing
   has to be arbitrated.

Mutation: with a chance per cell and update, one random byte of the cell is replaced (1 / 2^(n+1) at mutation
exponent n; a unit convention chosen to sit near the pair worlds' dial).

The random order of neighbours is one of the eight symmetries of the square, drawn for every cell and update,
so the rule has no preferred direction. The lattice wraps.

Code: `src/lib/gpu/around.ts`, engine `setupAround` and `simulateUpdates`, reference
`src/lib/dev/aroundReference.ts`. Z80 only, square lattice, 16-byte tapes.

## The two-layer law and running backwards

Every cell also remembers the tape it had before the last update (`before`). If `out` is the tape its run
produces from the lattice as it is now, the cell's next tape is

    now' = out xor now xor before,     before' = now.

`out xor now` is the change the run makes; the law applies it on top of the remembered tape. Because `out`
depends only on the present lattice and on the order of neighbours, an update can be undone exactly: exchange the
two layers and draw the orders in reverse (each cell's random counter is stepped back).

- **Dissipation** (the existing dial, 0 to 1) is the chance, per cell and update, that a cell forgets first
  (`before := now`). A cell that has forgotten gets `now' = out`, the plain rule. So dissipation 1 is the plain
  rule above, and dissipation 0 is a closed, reversible world.
- **Mutation** is written into `now'` only, so it is heating; with mutation on, running backwards does not
  retrace.
- **Running backwards** (`engine.reverseTime()`, a button in the app) exchanges the layers and reverses the
  draws. It stops at the update where the rule began (the start of the world, or the moment the rule was chosen).

`cells` keeps the plain rule.

## Free-running cells (`interaction: 'cells'`)

The same rule spread over time. Each processor keeps its registers and executes one instruction per tick. It
reads its neighbours as they are at that tick, writes outside its own tape are dropped, and it starts afresh
(blank registers, new order of neighbours) when its cell is full, at HALT, or after `steps` instructions.

## What was checked

`node scripts/around-check.mjs`: 25 checks, all pass.

- **Against an independent reference.** The rule is written a second time in TypeScript with the third-party
  `z80-emulator`. One GPU update is compared with one reference update cell by cell, each comparison starting
  from the GPU's own state, for both rules, with 1, 2 and 4 neighbours gathered, with mutation off and on (twelve
  settings, 192 cells): new tape, and for `neighbourhood` the number of instructions and the reason the run
  ended, for `cells` the registers and the order of neighbours. About 42,000 cell updates compared for
  `neighbourhood` and 457,000 for `cells`, no mismatch.
- **What is left out.** Cells whose run fetched an instruction on which the two Z80s are known to differ: `DD`
  and `FD` prefixes, repeating block input and output, `PUSH AF` (it writes the two undocumented flag bits to
  memory) and `LD A,R`. On the last, `z80-emulator` returns the refresh register with bit 3 cleared (set to 13,
  it loads 7 where the count is 15); this showed up as one mismatch in 76,343 and was traced to that
  instruction. 5–14% of cell updates are left out under `neighbourhood`, at most 1% under `cells`.
- **The two-layer law against the reference.** The same comparison at dissipation 0.5 and 0, with 1 and 4
  neighbours gathered, mutation off and on (eight settings), now including each cell's remembered tape: 5,417 to
  6,959 cell updates compared per setting, no mismatch.
- **Forwards, then backwards.** A closed world (dissipation 0, mutation off, 64 × 48 cells, four gathered) run
  300 updates forwards and 300 backwards is back at its start: 0 bytes differ in either layer, and every random
  counter is back (48,949 bytes had changed on the way out). With dissipation 0.5 the same procedure leaves 93,915
  bytes unrestored, as it should.
- **In the app** (`scripts/around-app-reverse.mjs`): rule chosen at update 952, world closed, 912 updates
  forwards, then the button: it ran back to update 952 and stopped, and both layers were as they had been.
- **Repeatability.** Two runs of one seed give the same lattice under both rules.
- **Orders of neighbours.** 61,440 draws, 7,534 to 7,757 per order (chi-squared 5.22 on 7 degrees of freedom).

Not checked: an exact symmetry test (the order of neighbours is drawn from each cell's random counter, so a
turned world does not draw the turned orders; the reference comparison and the count above stand in for it);
the 0.2 core against the reference (the comparison uses 0.3 `zilog-nmos`; the app's default is 0.2).

## First look: the neighbourhood rule

Random start, Z80 0.2 core, mutation exponent 4, `steps` 128, headless. Data: `neighbourhood/runs/`.

- **Patterns appear in every run, within a few hundred updates.** 160 × 126 cells, five seeds each
  (`s_g4_s*`, `s_g1_s*`), state at 1,000 and 4,000 updates:
  - four neighbours gathered: in four of five seeds one tape, `e1 e3` repeated (POP HL; EX (SP),HL), holds
    93.5–94.3% of cells; in the fifth (seed 4) the lattice is 63–64% blank bytes with `d5 11` pairs at many
    offsets and 6,400–6,500 distinct tapes, and stays so to 4,000 updates;
  - one neighbour gathered: `e1 e3` holds 82.0–85.5% of cells in all five.
- **Growth is round.** In `p_g4_s1` the replicator starts in one place near 40 updates and spreads as a disc: 1%
  of cells at 40 updates, 8% at 80, 19% at 120, 62% at 200, 94% at 300 (`neighbourhood/growth_four.png`). A second kind of patch (not read) grows beside it; by 300 updates the first holds 94% of cells. With one neighbour gathered it starts in
  several places (`growth_one.png`).
- **No direction is favoured in the numbers either.** The share of cells holding the same tape as their eastern
  neighbour and as their southern neighbour differ by at most 0.003 in all ten runs (for example 0.8892 and 0.8884).
- **Speed.** 3,000 updates of 9,216 cells took 1.0–1.5 s headless. In the app at 8× it showed 482 updates a
  second: the takeover is over in under a second, so use 1× to watch it.

## First look: the dissipation dial

One run per setting: 96 × 96 cells, seed 1, mutation exponent 4, 3,000 updates (`r_g*_d*`). "Replicator" is the
share of cells holding `e1 e3` repeated; "heat" is the share of bytes that differ between a cell's tape and the
tape it remembers.

| dissipation | 1 gathered: replicator | heat | 4 gathered: replicator | heat |
| ----------- | ---------------------- | ---- | ---------------------- | ---- |
| 0           | 0%                     | 100% | 0%                     | 100% |
| 0.1         | 0%                     | 42%  | 0%                     | 71%  |
| 0.3         | 14%                    | 54%  | 35%                    | 46%  |
| 0.5         | 30%                    | 49%  | 39%                    | 45%  |
| 0.7         | 39%                    | 46%  | 47%                    | 44%  |
| 0.9         | 51%                    | 43%  | 53%                    | 42%  |
| 0.95        | 55%                    | 42%  | 57%                    | 42%  |
| 0.99        | 80%                    | 26%  | 94%                    | 6%   |
| 1           | 84%                    | 21%  | 0% (blank-byte state)  | 39%  |

- **A closed world is indistinguishable from random bytes** (entropy 8.0 bits, 99.6% of bytes differ between the
  layers), as in the pair soup.
- **Between 0.1 and 0.3 the replicator appears, and its share then rises with dissipation** in both columns,
  without a jump until the last step to 0.99. At intermediate settings the replicator holds a steady part of the
  lattice among cells that are not it (`neighbourhood/dial_one.png`); its share at 500 and at 3,000 updates is
  about the same (for example 34.7% and 34.8% at dissipation 0.3 with four gathered).
- **The pair soup did not show this middle range** in its pilot (0 of 10 runs taken over at dissipation 0.3;
  `docs/reversible-phase-pilot.md`). The two differ in more than one way, so this is an observation, not a
  comparison.
- One run per setting. The blank-byte outcome at dissipation 1 with four gathered is the alternative state seen
  in one seed of five in the plain rule.

Unlike the pair engine, this rule is repeatable for a seed and a 3,000-update run takes about a second, so a
replicated map of dissipation against mutation can be run locally.

## How long to run

The author's question. Three things end a run, and the third is the only number:

1. **The cell is full**: the run has written all 16 bytes. Once a replicator holds the lattice this ends 97–99%
   of runs.
2. **HALT.**
3. **A cut-off**, for programs caught in a loop. From a random start it ends most runs (83% at 200 updates with
   four neighbours gathered); in the living state 0.5–3%.

One seed per setting, 96 × 96 cells, 3,000 updates (`b_g*_st*`):

| cut-off | 1 gathered (ring 32) | 2 gathered (ring 64) | 4 gathered (ring 128)    |
| ------- | -------------------- | -------------------- | ------------------------ |
| 16      | no pattern           | no pattern           | no pattern               |
| 32      | `e1 e3` 85% of cells | no pattern           | no pattern               |
| 64      | `e1 e3` 85%          | `e1 e3` 92%          | no pattern               |
| 128     | `e1 e3` 84%          | `e1 e3` 91%          | blank bytes with `d5 11` |
| 512     | `e1 e3` 85%          | `e1 e3` 90%          | blank bytes with `c5 01` |
| 2,048   | `e1 e3` 85%          | `e1 e3` 90%          | `e1 e3` 94%              |

- **Below the size of the ring nothing appears; from the size of the ring upwards something does**, in all three
  columns. So a natural cut-off needs no free number: one instruction per byte of memory (32, 64 or 128).
- **Above it the cut-off hardly matters** with one or two neighbours gathered (84–85% and 90–92% across the range). With four gathered the two kinds of outcome seen are the two also seen across seeds at 128, so one
  seed per setting cannot say whether the cut-off chooses between them.

## First look: free-running cells

96 × 96 cells, one seed per setting, 200,000 ticks (`c_*`, `cf_*`, `cb_*`). No pattern appeared in any of six runs:

| what ends a run               | 1 gathered                  | 4 gathered           |
| ----------------------------- | --------------------------- | -------------------- |
| HALT only                     | none; mean run 2,022 instr. | none; mean run 1,600 |
| HALT or full cell             | none; mean run 1,941        | none; mean run 1,286 |
| HALT, full cell or 128 instr. | none; mean run 113          | none; mean run 111   |

So the version I proposed (no cut-off) does nothing, and adding the cut-off does not rescue it. What differs
from the neighbourhood rule is that a run does not work on a private copy of its neighbourhood: the tapes it
reads change under it, and what it writes outside its cell is dropped where the neighbourhood rule lets the
program read it back. Which of the two matters has not been tested.

## Known gaps

- One replicator takes the lattice in a few hundred updates and holds it to 4,000. Whether anything follows it
  on longer runs has not been looked at; nor have other mutation levels, other lattice sizes, the second kind of
  patch, or the blank-byte state of seed 4.
- The cut-off is the `steps` setting (128), not yet set from the size of the ring.
- No measure of heredity; "replicator" means a tape that spreads.
- The app's Steps, pair-attempt and lethal-zero controls stay visible; only Steps is used. Checkpoints do not
  store the processors (only `cells` needs them).
- The dissipation dial has one run per setting, at one mutation level and one lattice size, to 3,000 updates.
- Running backwards retraces only while nothing else is changed; the app does not prevent a dial being moved
  between the forward and the backward run. Checkpoints do not store the direction of time.
