# Stage M — the 8080 subset (2026-10-09; pre-registered in `REVISION_PREREG.md`, M)

The Z80 restricted to the Intel 8080 instruction set by suppression (`i8080` in `make_conds.ABLATIONS`: the 256 CB-page
opcodes, the 78 defined ED-page opcodes, and EX AF,AF', DJNZ, the five JR forms and EXX; 342 instructions in all; the
literal word `01 c5` and the return design's `ad e3 21 c0` untouched, verified from the condition file). Stage G
conditions at L = 16 (300,000 steps) and L = 32 (one million), 20 worlds each, seeds 7001–7020, Modal batch `stageM`.
Scored by the Stage G pipeline (`stage_pipeline.sh`); tables in `stageM/`, scoring in `SCORING.md`.

| prediction | outcome |
|---|---|
| M1 first replicator a load–push word in ≥ 18/20 at both lengths | **met**: 20/20 and 20/20 (`01 c5` 13 + 14, `21 e5` 6 + 2, `11 d5` 1 + 4); medians 650 and 325 steps |
| M2 closed successor at L = 16 in ≥ 15/20, by a return design, none by block copy | **met**: 20/20, every one the byte-identical `ad e3 21 e3 21 c0 ad c0` (the Z80's modal closer at this length, which is pure 8080 code), copies 1.00, self-damage 0.00 |
| M3 closure at L = 32 in ≤ 5/20 by one million steps | **met**: 0/20; the open pusher dominates every world to the end |
| M4 the published negative result for long 8080 tapes is explained | supported: the only 8080 closer is the return design, whose popped addresses alias into the organism on a 32-byte ring (0xe321 mod 32 = 1) and into the partner on a 64-byte ring (mod 64 = 33); with no block copy and no relative jump, nothing else closes |

Reading. A second instruction set, a real one, gives the same beginning everywhere (open, the literal word) and shows what
closure needs: a closed design the instruction set can express at that ring size. The 8080 subset has one, the return,
and it works only on the short ring; so life at L = 16 goes open → closed exactly as in the Z80, and life at L = 32 stays
open for a million steps. The order of events is the machine family's; the supply of closers is the instruction set's.
