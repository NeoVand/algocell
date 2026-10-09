# Replicator zoo

Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.

## i8080@closure · L=16 · 128 steps · mutation 1/2^4
**first** (20 seeds)
- 20× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` — [LD rr,nn ; PUSH rr] ×8

**final** (20 seeds)
- 20× `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` — [EX (SP),rr ; LD rr,nn ; RET NZ ; XOR L ; RET NZ ; XOR L] ×2

## i8080@closure1M · L=32 · 128 steps · mutation 1/2^4
**first** (20 seeds)
- 20× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …` — [LD rr,nn ; PUSH rr] ×16

**final** (20 seeds)
- 20× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …` — [LD rr,nn ; PUSH rr] ×16
