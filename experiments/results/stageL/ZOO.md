# Replicator zoo

Design signatures of the first heritable replicator (`first`, the t_rep tape) and of the final tape when it is a faithful replicator against random partners (`final`). Tiled tapes are shown as one period ×n. Counts are seeds.

## none@closure10M · L=16 · 128 steps · mutation 1/2^4
**first** (10 seeds)
- 10× `21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5` — [LD rr,nn ; PUSH rr] ×8

**final** (10 seeds)
- 9× `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` — [EX (SP),rr ; LD rr,nn ; RET NZ ; XOR L ; RET NZ ; XOR L] ×2
- 1× `6b f3 cb db 9c ed b0 b3 6b f3 cb db 9c ed b0 b3` — [DI ; SET 3,E ; SBC A,H ; LDIR ; OR E ; LD r,r] ×2

## none@closure10M · L=20 · 128 steps · mutation 1/2^4
**first** (10 seeds)
- 10× `21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5` — [LD rr,nn ; PUSH rr] ×10

**final** (10 seeds)
- 8× `1e a4 ed b0 1e a4 ed b0 1e a4 ed b0 1e a4 ed b0 1e a4 ed b0` — [LD r,n ; LDIR] ×5
- 1× `11 44 6b ed b8 94 be 5a 11 44 6b ed b8 94 be 5a 11 44 6b ed` — [CP (HL) ; LD r,r ; LD rr,nn ; LDDR ; SUB H] ×2+4B
- 1× `5d ce 65 11 e6 26 30 ed b8 11 5d ce 65 11 e6 26 30 ed b8 11` — [CP B ; LD rr,nn ; LD r,r ; LD rr,nn ; JR NC,d] ×2
