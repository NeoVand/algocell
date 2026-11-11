# The free-running world

11 November 2026, third design (the facing rule; see "A direction was built in"). A first look, not a
result: one to five runs per setting.

## The rule

A third value of the interaction setting, `free`. It removes three things the pair worlds set by hand: the step
budget (128), the cell (16 bytes) as the unit that runs, and the list of who meets whom.

1. **The world is a sheet of bytes and nothing else**, wrapped as a torus. The 16-byte tiles are only how the
   renderer draws it. Any width and height.
2. **A processor lands on a random byte, facing a random way, with every register at zero**, and starts
   executing there.
3. **Addresses are measured from where it landed, in its own frame.** A Z80 address is two bytes: the low byte
   counts bytes along the way the processor faces, the high byte counts lines across, each read as a signed offset
   (−128…127). Address 0 is the landing byte; the program counter walks forward; the stack starts at address 0,
   so the first push writes on the neighbouring line just behind the landing byte, and the stack grows backwards
   along that line. "Facing" is one of the eight symmetries of the square: forward is one of four directions and
   the stack is on either side. A processor can reach a 256 × 256 window around its landing byte and nothing
   outside it.
4. **One tick is one instruction for every processor**, all at once.
5. **A processor that executes HALT lifts off and lands somewhere else** (rule 2). Nothing else ends a run.

Mutation: with a small chance per instruction, one random byte anywhere on the sheet is replaced.

What is left to choose: how many processors there are (`heads`, per tile), and the mutation dial. Three optional
switches exist for comparison: `facing` (`any` as above; `same`: every processor reads rightwards with its stack
on the row above, which was the rule until this revision; `rows`: rightwards or leftwards, stack above or below;
`two-axes`: rightwards or downwards), `haltRule` (`restart` as above; `wait`: the processor stays on the HALT byte until
something changes it; `pass`: it steps over it) and `turnover` (a chance per instruction of lifting off without a
HALT; 0 by default).

The mutation dial counts in units of 128 instructions so that one position means roughly the same heat in all three
worlds: at exponent _n_ the chance per instruction is 1 / (128 × 2^_n_). This is a unit, not a rule.

A tick is a function of the sheet and the processors. Every processor reads the sheet as it stood when the tick
began; writes (and mutations) go to a log; each byte goes to the lowest-numbered processor that wanted it. All
randomness comes from a counter each processor carries. Nothing depends on GPU timing, so a seed gives the same
world every time, and many ticks can be queued at once.

Code: `src/lib/gpu/free.ts` (shader and starting processors), `src/lib/gpu/engine.ts` (`setupFree`,
`simulateTicks`, `readHeads`), reference `src/lib/dev/freeReference.ts`. Z80 only, square lattice, 16-byte tiles.

## A direction was built in (the author's objection to the second design)

The second design (commit c5ff5fa) was this rule with every processor facing the same way: the program counter
always walked right and the stack was always the row above. The pictures were full of horizontal lines, and the
author objected that the rule broke the symmetry of space. That was correct, and it was not cosmetic.

- **The asymmetry was in the rule.** In the second design a byte equals the byte two places along its row 94% of
  the time after a takeover and the byte two places down its column 53% of the time (chance level); in the waiting
  state, 59–64% against 24–29% (seeds 1 and 2). Under `any` both are 36–37% (`free-running/free_facing.png`).
- **The rule now commutes with every symmetry of the square, exactly**, and landing processors face the eight
  ways equally often (both under "What was checked").
- **The replicators of the second design needed the built-in direction.** Five seeds per rule, 100,000 ticks,
  otherwise the defaults of "First look" below (`sym_f_*`, `sym_a_seed*`):

| facing                                       | left the waiting state by 100,000 ticks | by                                   |
| -------------------------------------------- | --------------------------------------- | ------------------------------------ |
| `same` (rightwards; the second design)       | 5 of 5                                  | `01 c5`, `11 d5`, looping, `fc` rows |
| `rows` (right or left, stack above or below) | 5 of 5                                  | `01 c5` in all five                  |
| `two-axes` (rightwards or downwards)         | 0 of 5                                  |                                      |
| `any` (eight ways; the rule now)             | 0 of 5                                  |                                      |

The `rows` and `two-axes` arms were run to test a prediction written down before running them: a pusher row
reads the same forwards and backwards, so what should matter is whether all processors share one axis, not
whether they share a direction along it or a side for the stack. Both arms came out as predicted. The `same`
arm reproduces the second design's five runs.

- **Under `any`, nothing else was seen either**, in the few settings tried (seed 1 and 2 each): mutation off and
  mutation exponent 8 to 300,000 ticks, exponent 4 to 1,000,000 ticks. All stay in the waiting state.
- **A pusher world does not survive mutation under `any`.** A sheet that is a chessboard of `01` and `c5` (every
  row and every column reads `01 c5 01 c5 …`), or rows of `01 c5`, stays a pusher world for 50,000 ticks with
  mutation off (`c5` 81–83% of bytes, no processor halts). With mutation it falls to the waiting state: by 10,000
  ticks at exponent 4, by 20,000 at exponent 8, between 100,000 and 300,000 at exponent 12 (one run each). A
  64 × 64-byte patch of either pattern placed in a random sheet is gone by 2,000 ticks (one run each); under
  `same` the same patch spreads (`c5` 40% of bytes at 2,000 ticks).
- **So the waiting state is the winner here, not an absence.** In it 40.8–42.0% of processors are about to
  execute a blank byte and 3.3–3.5% an `ff` (three seeds at 100,000 ticks, `runs/sym_waiting.jsonl`); 136–160 of
  the 4,096 processors sit exactly at address 0x38, one address in 65,536. `ff` is "restart at 38": push the
  address of the next byte and jump to address 0x38. The commonest non-blank bytes are `39`, `3a`, `3b`, `3c` in
  that order in all five seeds. A reading that fits all three facts, not traced on the machine: a processor that
  meets an `ff` a few bytes after address 0x38 jumps back to 0x38, slides over blank bytes to the same `ff`, and
  writes the return address (`39 00`, `3a 00`, …) further along its stack each time round. It copies nothing;
  it writes blanks and small numbers, which are slides and landing strips for the next processor.

What this does and does not say. In this world, with this chip, at one density and three mutation levels, the
patterns that copy themselves need every reader to share one axis. It does not say that no heredity is possible
under the symmetric rule: the runs are short and few, and nothing here lets a pattern tell a processor which way
to read it. In the two-cell soups and in BFF the shared axis is given silently (a tape has a first byte and a
direction).

## Why the first design looked dead

The first design (commit 32f7fcb) tied each processor to one row or one column and laid its 64K address space
along that line. A processor could only ever write on its own line, so nothing could spread across the sheet, and
every row became a separate small world. That is what the first pictures showed.

I also suspected that never resetting a processor was part of the problem. It is not: with the new address map,
the `wait` and `pass` rules (no processor is ever reset) still grow replicators (table below).

## What was checked

`node scripts/free-check.mjs`: 19 checks, all pass.

- **Against an independent reference.** The rule is written a second time in TypeScript with the third-party
  `z80-emulator` executing instructions. Sheet bytes, registers, landing sites, facings and the number of restarts
  agree at every fifth tick for as long as the comparison is meaningful, in fourteen runs: all three HALT rules on
  a 32 × 32-byte sheet with 64 processors facing all eight ways (15 to 315 ticks), two runs with every processor
  facing the same way (65 ticks each), and the restart rule with a restart chance of 1/64 and mutation on, on a
  280 × 36-byte sheet (wider than a window, sides not powers of two; 45 to 135 ticks). 1,090 ticks in all; ten of
  the fourteen lasted 40 ticks or more and included bytes that two processors wanted in the same tick.
- **Why a comparison stops.** It ends at the first tick on which a processor fetches a `DD` or `FD` prefix,
  because Zilion and `z80-emulator` split prefix chains into steps differently, or a repeating block input or
  output instruction (`ED B2`, `B3`, `BA`, `BB`). On those the two disagree about two flags (H and P/V): this
  showed up as four failed comparisons when the facings changed which instructions the test runs met. Zilion's
  documentation says its 0.3 core follows the hardware test vectors there; I did not test which of the two is
  right. Bits 3 and 5 of the flag register (undocumented) are not compared.
- **No preferred direction.** With the `pass` rule and no restart chance or mutation a tick uses no random
  numbers. On a 96 × 96-byte sheet with 576 processors, turning or mirroring the whole world (sheet, landing
  sites, facings) by each of the seven non-trivial symmetries of the square and running 600 ticks gives the turned
  or mirrored result: 0 differing bytes and 0 differing register words in all seven (9,071 bytes changed by the
  run).
- **Facings are drawn evenly.** 102,400 landings: 12,717 to 12,889 per facing (chi-squared 2.87 on 7 degrees of
  freedom); landing sites are even across the quarters of each axis (1.09 and 9.45 on 3).
- **Repeatability.** Two runs of one seed with 4,096 processors, 500 ticks, restart chance and mutation on give
  identical sheets (63,490 bytes changed by the run, 0 differ).
- **A known replicator works where nothing competes.** On a sheet of HALT bytes with one 512-byte block of the
  pair worlds' pusher (`01 c5`), the count of pusher bytes rises at every mark over 2,000 ticks (512 → 20,796
  under `any`; it was 512 → 4,503 under the second design).

Not checked: more than 360 ticks against the reference; the 0.2 `classic` core against the reference (the
comparison uses 0.3 `zilog-nmos`; the app's default is 0.2); whether an instruction ever needs more distinct
writes than the log holds (five entries; the Z80 should need two, plus one for a mutation).

## First look under the second design (`facing: same`)

Everything in this section was run before the facing rule existed, with every processor reading rightwards. It
describes that rule, which builds a direction in; under `any` none of it has been seen. The five seeds of part 1
were run again as `facing: same` after the change and gave the same outcomes.

All runs: 512 × 512 bytes, random start, Z80 0.2 core, headless. Unless stated: 4,096 processors (1 per 4 tiles),
HALT restarts, restart chance 0, mutation exponent 4 (1 byte per 2,048 instructions). Data: `free-running/runs/`,
one file per run as written by `scripts/free-look.mjs`; `node docs/free-running/summarise.mjs <files>` prints one
line per saved tick.

### 1. Replicators appear by themselves

Five seeds, 100,000 ticks each (`b_seed1` … `b_seed5`).

| seed | waiting state until     | then                       | by 100,000 ticks                        |
| ---- | ----------------------- | -------------------------- | --------------------------------------- |
| 1    | about 3,500             | `01 c5`                    | `01 c5`                                 |
| 2    | between 50,000 and 100k | `01 c5`                    | `01 c5`                                 |
| 3    | between 20,000 and 50k  | `01 c5`                    | `c5 01 c5 01 18 f8` (from about 65,000) |
| 4    | between 10,000 and 20k  | `11 d5`                    | `d5 11 d5 11 18 f8`                     |
| 5    | between 10,000 and 20k  | rows of `fc xx` (not read) | rows of `fc xx`                         |

- **The waiting state** has 40–43% zero bytes and many `xx 00` pairs (`39 00` the most common). The pair worlds'
  waiting state has 39–40% zero bytes (pilot note for the reversible rule). Unexplained; one guess is that the
  pairs are return addresses pushed by calls and restarts, whose high byte is 0 in the landing row. Not tested.
- **`01 c5` and `11 d5` are the pushers of the pair worlds** (load a register pair with the two bytes, push it).
  Here they need no cell and no budget: a processor running along a row of pushers writes pushers along the row
  above.
- **The takeover is fast**: in `g_pic_s1`, `c5` goes from 1% of bytes at 3,000 ticks to 56% at 6,000.
- Seed 5's world of `fc xx` rows has not been read. `fc` is a conditional call, so those rows may be made of pushed
  addresses.

### 2. A second takeover: a variant that keeps its processor

In seeds 3 and 4 a second pattern replaces the pusher: the pusher plus `18 f8`, a jump back by six bytes. The
repeating unit becomes six bytes (a byte equals the byte six places to its right 75% of the time). Both runs stay in
that state to 1,000,000 ticks (`e_long_s3`, `e_long_s4`).

Traced by hand on one arrangement of the bytes, not on the machine: where a stretch of pushers meets `18 f8`, a
processor pushes, reaches the jump, lands six bytes back and pushes again, for ever, and what it pushes is the
six-byte unit including the jump. A plain pusher's processor runs on along its row and is lost when the row ends.

An invasion test, mutation off, three seeds each, 100,000 ticks (`f_*`):

| start                                                          | bytes `18` at 2,000 ticks | at 100,000 ticks |
| -------------------------------------------------------------- | ------------------------- | ---------------- |
| sheet of `01 c5`, one 512-byte block of `c5 01 c5 01 18 f8`    | 17.2%, 17.8%, 17.9%       | 22.1–24.3%       |
| sheet of `01 c5`, one 512-byte block of `11 d5` (other pusher) | 0 (the block dies out)    | 0                |
| sheet of `01 c5` only                                          | 0                         | 0                |

The looping pattern invades from 0.2% of the sheet in every run; a block of the other pusher is lost in every run;
without mutation a pusher world never makes `18 f8` by itself.

The same variant appeared with mutation off (seed 1), with the `wait` rule (seed 2), and, using `10 f8` (a counted
jump) instead of `18 f8`, with a restart chance of 1/1,024 (seed 1).

At the app's default size (576 × 576 bytes here, 5,184 processors; `h_app_size_s6` … `s8`, 600,000 ticks) all
three seeds end in the looping variant, by 200,000 ticks in two and by 300,000 in the third, which first spent
200,000 ticks in an `fc` world like seed 5's. One live test in the app was still in the waiting state after 274,000
ticks.

### 3. Run length, now that nothing sets it

Under the restart rule with no restart chance, the length of a finished run in the waiting state has a median of
about 2,500–3,800 instructions (seeds 2–4 between 10,000 and 50,000 ticks). After a takeover processors almost
stop halting: the average run length at 1,000,000 ticks is 172,529 and 206,618 instructions.

### 4. The dials (two seeds each, 100,000 ticks unless stated)

| change from the default                      | seed 1                             | seed 2                     |
| -------------------------------------------- | ---------------------------------- | -------------------------- |
| mutation off                                 | `01 c5` by 50,000, looping by 100k | `01 c5` by 20,000          |
| mutation exponent 8 (16 times less)          | `01 c5` by 50,000                  | `01 c5` by 20,000          |
| mutation exponent 0 (16 times more)          | waiting throughout (39% zero)      | waiting throughout         |
| restart chance 1 in 1,024                    | `11 d5`, then looping with `10 f8` | `01 c5` by 50,000          |
| restart chance 1 in 128                      | `01 c5` by 50,000                  | waiting throughout         |
| HALT: wait                                   | `01 c5` by 50,000                  | looping variant by 100,000 |
| HALT: pass                                   | waiting throughout                 | `01 c5` by 100,000         |
| 1,024 processors (1 per 16 tiles)            | `01 c5` by 50,000                  | waiting throughout         |
| 16,384 processors (1 per tile), 50,000 ticks | waiting throughout                 | waiting throughout         |

Two runs per row say little. What can be said: replicators appear with mutation off; they did not appear at the
highest mutation tried; no HALT rule is required for them. Rows with different numbers of processors are not equal
amounts of work per tick.

### Speed

1,000,000 ticks of 4,096 processors took 15.3 s headless: about 270 million instructions a second. In the app at
speed 8× the default world (160 × 126 tiles, 5,040 processors) showed 15,340 ticks a second.

## Pictures

- `free-running/free_facing.png`: one starting sheet under the four facing rules at 100,000 ticks
  (`figures_facing.py`).
- The three below show the second design (`facing: same`).
- `free-running/free_takeover.png`: the sheet at 2,000, 4,000, 4,500 and 5,500 ticks (seed 1).
- `free-running/free_closeups.png`: 32 × 10 bytes written out, with processors outlined: waiting state, pushers,
  looping variant.
- `free-running/in_the_app.png`: the app 18 s after choosing Free-running.
- `free-running/figures.py` draws the first two from the saved sheets (`scripts/free-look.mjs` writes them).

## Known gaps

- Under the symmetric rule the world has only been seen in its waiting state. Not tried: other processor
  densities, the `wait` and `pass` rules, a restart chance, sheets of other sizes, runs beyond 1,000,000 ticks,
  the 0.3 core, and any rule by which the bytes themselves set the way a landing processor faces.
- The reading of the waiting state (restart-at-38 loops) rests on counts, not on a trace.
- No view of the processors in the app. The close-up picture is drawn offline.
- No measure of heredity. "Replicator" here means a pattern that spreads and that, seeded as a block, invades.
- The app's counter and rate are in ticks but still labelled "Epoch" and "updates/s"; the takeover is over in
  seconds at 8×, so use 1× to watch it.
- Steps, pair attempts, lethal zero and edges stay visible in the instrument and do nothing here.
- Checkpoints do not store the processors.
- The reversible law is not combined with this world. Its sheet could obey the two-layer law, but the processors'
  registers forget, so the world as a whole would not be reversible without more thought.
- Ties between two processors go to the lower number, a small fixed bias.
