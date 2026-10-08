# Replicator zoo

Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.

## lethal@closure · L=16 · 128 steps · mutation 1/2^4
**first** (10 seeds)
- 1× `84 5e ed b0 84 5e ed b0 84 5e ed b0 84 5e ed b0` — [ADD A,H ; LD r,(HL) ; LDIR] ×4
- 1× `a8 6e eb a0 ed b0 28 56 a8 6e eb a0 ed b0 28 56` — [AND B ; LDIR ; JR Z,d ; XOR B ; LD r,(HL) ; EX rr,rr] ×2
- 1× `31 a4 02 d1 e8 9e ed b0 31 a4 02 d1 e8 9e ed b0` — [LD rr,nn ; POP rr ; RET PE ; SBC A,(HL) ; LDIR] ×2
- 1× `a4 5e ed b0 a4 5e ed b0 a4 5e ed b0 a4 5e ed b0` — [AND H ; LD r,(HL) ; LDIR] ×4
- 1× `b7 11 88 d9 69 ed b0 a0 b7 11 88 d9 69 ed b0 a0` — [AND B ; OR A ; LD rr,nn ; LD r,r ; LDIR] ×2
- 1× `11 68 4f ed b0 9b 11 dc 11 68 4f ed b0 9b 11 dc` — [LD r,r ; LD r,r ; LDIR ; SBC A,E ; LD rr,nn] ×2

**final** (10 seeds)
- 2× `44 5e ed b0 44 5e ed b0 44 5e ed b0 44 5e ed b0` — [LD r,(HL) ; LDIR ; LD r,r] ×4
- 1× `c8 6e eb ed b0 ee b6 4e c8 6e eb ed b0 ee b6 4e` — [EX rr,rr ; LDIR ; XOR n ; LD r,(HL) ; RET Z ; LD r,(HL)] ×2
- 1× `11 48 da ed b0 44 c8 11 11 48 da ed b0 44 c8 11` — [JP C,nn ; LD r,r ; RET Z ; LD rr,nn] ×2
- 1× `a4 5e ed b0 a4 5e ed b0 a4 5e ed b0 a4 5e ed b0` — [AND H ; LD r,(HL) ; LDIR] ×4
- 1× `11 e8 da ed b0 58 6e 9b 11 e8 da ed b0 58 6e 9b` — [LD r,(HL) ; SBC A,E ; LD rr,nn ; LDIR ; LD r,r] ×2
- 1× `11 c8 b7 ed b0 4a 0c 77 11 c8 b7 ed b0 4a 0c 77` — [INC C ; LD (HL),r ; LD rr,nn ; LDIR ; LD r,r] ×2
