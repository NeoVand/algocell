# Replicator zoo

Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.

## mixed@closure · L=16 · 128 steps · mutation 1/2^4
**first** (10 seeds)
- 10× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` — [LD rr,nn ; PUSH rr] ×8

**final** (10 seeds)
- 5× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` — [LD rr,nn ; PUSH rr] ×8
- 3× `b5 e3 21 e3 21 e0 b5 e0 b5 e3 21 e3 21 e0 b5 e0` — [EX (SP),rr ; LD rr,nn ; RET PO ; OR L ; RET PO ; OR L] ×2
- 1× `bd e3 21 e3 21 c0 bd c0 bd e3 21 e3 21 c0 bd c0` — [CP L ; EX (SP),rr ; LD rr,nn ; RET NZ ; CP L ; RET NZ] ×2
- 1× `b0 5e c2 30 eb ed b0 9c f3 d0 95 d6 00 36 14 60` — LD r,(HL) ; JP NZ,nn ; LDIR ; RET NC ; LD (HL),n
