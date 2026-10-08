# Notes for the three.js figures: what must be exact, and the exact traces (2026-10-08)

The draft "Heredity before individuality" has the right idea and the right mood. Before it is rebuilt exactly, four
facts in it need correcting, because the mechanism is the message.

## Corrections of fact

1. **`01 c5` does not "copy the next two bytes to the next address".** `01` is `LD BC,nn`: it loads the two bytes that
   follow it (`c5 01`) into a register pair. `c5` is `PUSH BC`: it writes those two bytes at the **stack pointer**, which
   starts at the far end of the partner's tape (cell 31 of the 32-byte ring) and moves two cells towards lower addresses
   with every push. So the copies are not written just ahead of the executing instruction; the read head (IP) moves right
   from the organism's first byte and the write head (SP) moves left from the partner's last byte. They approach each
   other. The copy lands in the partner, in phase with the organism, because both the organism's period and the push are
   two bytes.
2. **The "growing run of identical code" is written from the far end backwards**, not extended forwards. If the figure
   keeps a single long tape (a fine simplification), the writes should appear at the right end and grow leftwards while
   the executing position moves rightwards.
3. **The closer is not "a third byte c5 that jumps back".** `c5` is a push and cannot jump. The real closers are `RET NZ`
   (`c0`, which returns to an address formed from the organism's own written bytes), `JR NZ` (`20 f0`, a relative jump),
   `DJNZ` (`10 e5`, a counted jump) and `LDIR` (`ed b0`, a block copy that repeats in place). The L = 16 closed successor
   that took over in 17 of 20 worlds is `ad e3 21 e3 21 c0 ad c0` repeated twice; its jump is the `c0`. If the figure
   needs one closer, use that tape and the `c0`, and draw the return as execution coming back **inside the organism**,
   not as a jump along the long tape.
4. **The memory is two tapes in one ring.** Organism A occupies cells 0–15, partner B cells 16–31; the IP starts at 0, the
   SP at 31; after 128 instructions both tapes are written back. "Occupied memory" to the left of the organism does not
   exist in an encounter; what lies beyond the organism is the partner, and the point of the open phase is that execution
   runs into it.

The tagline "a sequence that makes a copy of itself is already alive to tomorrow" is lovely and should stay. The
caption "individuality is a return in control flow" is exactly right.

## Exact traces (one encounter against an all-zero partner, 128-instruction budget; from the GPU kernel)

Cells are numbered 0–31. "IP before" is the cell the instruction is fetched from; "SP before" is the stack pointer
before the instruction; a push writes cells SP−2 and SP−1 and then SP decreases by two. The partner being all zeros, every
instruction fetched beyond cell 15 is a `NOP` (zero), so the open pusher simply runs to the end of the budget.

### The first replicator, L = 16: `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5`

| instruction | IP before | SP before | bytes written into B so far | note |
|---|---|---|---|---|
| 0 | 0 | 31 | 0 |  |
| 1 | 3 | 31 | 0 |  |
| 2 | 4 | 29 | 2 | push wrote cells 29, 30 |
| 3 | 7 | 29 | 2 |  |
| 4 | 8 | 27 | 4 | push wrote cells 27, 28 |
| 5 | 11 | 27 | 4 |  |
| 6 | 12 | 25 | 6 | push wrote cells 25, 26 |
| 7 | 15 | 25 | 6 |  |
| 8 | 16 | 23 | 8 | push wrote cells 23, 24; IP has left the organism (now in B) |
| 9 | 17 | 23 | 8 |  |
| 10 | 18 | 23 | 8 |  |
| 11 | 19 | 23 | 8 |  |
| 12 | 20 | 23 | 8 |  |
| 13 | 21 | 23 | 8 |  |
| 14 | 22 | 23 | 8 |  |
| 15 | 23 | 23 | 8 |  |
| 16 | 24 | 21 | 10 | push wrote cells 21, 22 |
| 17 | 27 | 21 | 10 |  |
| 18 | 28 | 19 | 12 | push wrote cells 19, 20 |
| 19 | 31 | 19 | 12 |  |
| 20 | 0 | 19 | 12 | jump back: a cell is revisited |
| 21 | 3 | 19 | 12 |  |
| 22 | 4 | 17 | 14 | push wrote cells 17, 18 |
| 23 | 7 | 17 | 14 |  |
| 24 | 8 | 15 | 15 | push wrote cells 15, 16 |

Reading: four `LD` + `PUSH` pairs execute the organism's 16 cells and write 8 bytes (cells 23–30); the pass over itself
leaves the copy half done; the IP then leaves the organism at instruction 8 and never returns. Against a random partner the
bytes it fetches there are executed, and the outcome depends on them.

### The closed successor, L = 16: `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0`

| instruction | IP before | SP before | bytes written into B so far | note |
|---|---|---|---|---|
| 0 | 0 | 31 | 0 |  |
| 1 | 1 | 31 | 0 |  |
| 2 | 2 | 31 | 1 | push wrote cells 31, 32 |
| 3 | 5 | 31 | 1 |  |
| 4 | 6 | 31 | 1 |  |
| 5 | 7 | 31 | 1 |  |
| 6 | 0 | 1 | 1 | jump back: a cell is revisited |
| 7 | 1 | 1 | 1 |  |
| 8 | 2 | 1 | 1 |  |
| 9 | 5 | 1 | 1 |  |
| 10 | 3 | 3 | 1 | jump back: a cell is revisited |
| 11 | 4 | 3 | 1 |  |
| 12 | 7 | 3 | 1 |  |
| 13 | 3 | 5 | 1 | jump back: a cell is revisited |
| 14 | 4 | 5 | 1 |  |
| 15 | 7 | 5 | 1 |  |
| 16 | 0 | 7 | 1 | jump back: a cell is revisited |
| 17 | 1 | 7 | 1 |  |
| 18 | 2 | 7 | 1 |  |
| 19 | 5 | 7 | 1 |  |
| 20 | 0 | 9 | 1 | jump back: a cell is revisited |
| 21 | 1 | 9 | 1 |  |
| 22 | 2 | 9 | 1 |  |
| 23 | 5 | 9 | 1 |  |
| 24 | 3 | 11 | 1 | jump back: a cell is revisited |
| 25 | 4 | 11 | 1 |  |
| 26 | 7 | 11 | 1 |  |
| 27 | 3 | 13 | 1 | jump back: a cell is revisited |
| 28 | 4 | 13 | 1 |  |
| 29 | 7 | 13 | 1 |  |
| 30 | 0 | 15 | 1 | jump back: a cell is revisited |
| 31 | 1 | 15 | 1 |  |
| 32 | 2 | 15 | 2 | push wrote cells 15, 16 |
| 33 | 5 | 15 | 2 |  |
| 34 | 0 | 17 | 2 | jump back: a cell is revisited |
| 35 | 1 | 17 | 2 |  |
| 36 | 2 | 17 | 4 | push wrote cells 17, 18 |
| 37 | 5 | 17 | 4 |  |
| 38 | 3 | 19 | 4 | jump back: a cell is revisited |
| 39 | 4 | 19 | 6 | push wrote cells 19, 20 |
| 40 | 7 | 19 | 6 |  |
| 41 | 3 | 21 | 6 | jump back: a cell is revisited |
| 42 | 4 | 21 | 8 | push wrote cells 21, 22 |
| 43 | 7 | 21 | 8 |  |
| 44 | 0 | 23 | 8 | jump back: a cell is revisited |
| 45 | 1 | 23 | 8 |  |
| 46 | 2 | 23 | 10 | push wrote cells 23, 24 |
| 47 | 5 | 23 | 10 |  |
| 48 | 0 | 25 | 10 | jump back: a cell is revisited |

Reading: the IP never exceeds cell 15; `c0` (`RET NZ`) pops a return address formed from the organism's own written bytes
and execution comes back inside; the stack pointer walks through the organism's own cells. (Full trace in
`results/concept/traces.json`; the other closers are there too: `jrnz_L50`, `djnz_L20`, `ldir_L20`.)

## What an exact three.js animation should show

- Two tapes end to end (A teal, B grey), 32 cells, as a ring or a straight run with the seam marked.
- Two moving heads: the IP (teal) stepping through A's cells; the SP (vermilion) starting at B's last cell and stepping
  left two cells per push, writing `c5 01` into B each time.
- The moment the IP crosses cell 15 into B (the open phase in one frame).
- For the closer: the IP turning back at `c0`, the stack pointer walking inside A, every B cell becoming a copy.
- Nothing else: no "occupied memory", no infinite tape, no third byte that jumps.
