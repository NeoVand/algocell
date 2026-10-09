# Replicator zoo

Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.

## none@closure1M · L=32 · 128 steps · mutation 1/2^4
**first** (20 seeds)
- 20× `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …` — [LD rr,nn ; PUSH rr] ×16

**final** (20 seeds)
- 8× `11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 …` — [LD rr,nn ; PUSH rr] ×16
- 6× `04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 …` — [INC B ; LD r,(HL) ; LDIR] ×8
- 3× `44 5e ed b0 44 5e ed b0 44 5e ed b0 44 5e ed b0 44 5e ed b0 …` — [LD r,(HL) ; LDIR ; LD r,r] ×8
- 2× `1e 04 ed b0 1e 04 ed b0 1e 04 ed b0 1e 04 ed b0 1e 04 ed b0 …` — [LD r,n ; LDIR] ×8
- 1× `84 5e ed b0 84 5e ed b0 84 5e ed b0 84 5e ed b0 84 5e ed b0 …` — [ADD A,H ; LD r,(HL) ; LDIR] ×8
