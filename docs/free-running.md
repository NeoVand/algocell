# The free-running world

12 November 2026. A first build to look at, not a result.

## What it is

A third value of the interaction setting, `free`. It removes three things the other worlds set by hand: the
step budget (128), the cell (16 bytes) as the unit that runs, and the reset of the processor at every encounter.

- **Space** is the lattice's bytes read as one continuous sheet, wrapped as a torus. The 16-byte tiles are only how
  the renderer draws it.
- **A head** is a processor that is never reset: registers and program counter persist. Each head lives on one line
  of the sheet (a row or a column, in one of the two senses) and its 16-bit address space is laid along that line
  from its origin. A head can read and write anywhere on its line and nowhere else. Rows and columns meet at every
  byte, which is how lines affect each other.
- **Time** is one instruction per head per tick. There are no encounters.
- **Heads are conserved.** Their number, origins and directions are a starting condition.
- **HALT** has no interrupt to end it. Under `wait` the head stays on the HALT byte and fetches it again every tick,
  so it sleeps until something else changes that byte. Under `pass` it moves on.
- **Mutation** replaces random bytes of the sheet: `floor(heads / 2^exponent)` bytes per tick.

A tick is a function of the sheet and the heads. Every head reads the sheet as it stood when the tick began;
writes go to a log; each byte goes to the lowest-numbered head that wrote it. Nothing depends on GPU timing, so a
seed gives the same world every time (the pair worlds do not have this property).

Settings: `heads` per tile (default 0.25), `headDirections` (`rows`, `weave` = rows and columns, `all`),
`haltRule` (`wait`, `pass`). Z80 only, square lattice, 16-byte tiles, width and height powers of two so that a line
divides the 16-bit address space.

Code: `src/lib/gpu/free.ts` (shader and starting heads), `src/lib/gpu/engine.ts` (`setupFree`,
`simulateFreeTick`, `readHeads`), reference `src/lib/dev/freeReference.ts`.

## What was checked

`node scripts/free-check.mjs`, all nine checks pass.

- **Against an independent reference.** The rule is written a second time in TypeScript with the third-party
  `z80-emulator` executing instructions. On a 32 × 32-byte sheet with 64 heads in all four directions, sheet bytes
  and registers agree at every tenth tick for as long as the comparison is meaningful: 210, 80, 170 and 150 ticks
  (rule `wait`, seeds 3–6) and 70, 80, 90 and 150 ticks (rule `pass`). Each run included bytes that two heads
  wanted in the same tick (3 to 11) and, under `wait`, up to 662 head-ticks spent on a HALT byte.
- **Why the comparison stops.** It ends at the first tick on which a head fetches a `DD` or `FD` prefix, because
  Zilion and `z80-emulator` split prefix chains into steps differently and the reference has a known deviation
  there. The starting sheets have those two bytes removed; the programs create them again within a few hundred
  ticks. Bits 3 and 5 of the flag register (undocumented, modelled differently by the two cores) are not compared.
- **Repeatability.** Two runs of one seed with 4,096 heads, 500 ticks and mutation on give identical sheets
  (64,136 bytes changed by the run, 0 differ).

Not checked: more than 210 ticks against the reference; the 0.2 `classic` core against the reference (the
comparison uses 0.3 `zilog-nmos`); mutation against the reference; whether a head ever needs more than four
distinct writes in one instruction (the log holds four; the Z80 should need two).

## First look (single runs, seed 6, 512 × 512 bytes)

| heads  | lines            | HALT | mutation    | ticks  | zero bytes | heads waiting on HALT | byte entropy, bits |
| ------ | ---------------- | ---- | ----------- | ------ | ---------- | --------------------- | ------------------ |
| 16,384 | rows and columns | wait | exponent 10 | 8,000  | 35.0%      | 25.8%                 | 5.53               |
| 1,024  | rows and columns | wait | none        | 30,000 | 20.2%      | 55.5%                 | 6.99               |
| 4,096  | rows only        | wait | none        | 30,000 | 32.9%      | 26.7%                 | 5.93               |
| 1,024  | rows and columns | pass | none        | 30,000 | 26.2%      | 0                     | 6.51               |

- The world does not freeze. Under `wait` a quarter to a half of the heads are parked on a HALT byte at any time
  and the rest keep the sheet changing.
- With rows only, every row becomes its own small world, and most settle into a short repeating pattern, each row a
  different one.
- With rows and columns the patterns cross and overwrite each other: a plaid at low head density, close to noise at
  one head per tile.
- Speed: 8,000 ticks of 16,384 heads took 0.86 s headless on a loaded machine, about 150 million instructions a
  second.

Pictures and the numbers behind the table are in `docs/free-running/` (a 160 × 160-byte corner of the sheet at
two times for the rows-only and sparse runs; the whole sheet for the dense run; one screenshot of the app).

Nothing here says whether anything replicates in the sense of the paper. There is no measure of heredity for this
world yet.

## Known gaps

- No view of where the heads are. The renderer shows only the bytes.
- The step budget, pair attempts, lethal zero and edges controls stay visible in the instrument but do nothing here.
- Lethal zero, ablation, random registers, the stack convention and the dials are ignored in this world.
- Checkpoints do not store the heads.
- The reversible law is not combined with it yet. A single instruction's read set and write set would have to be
  recorded to do that.
- A head reaches its whole line in one instruction (`LD (nn),A`), so along a line there is no speed limit. Only
  sequential execution and block copies are local.
