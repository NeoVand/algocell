# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 32 (none@closure1M, horizon 1,000,000, 20 worlds)

- worlds with a heritable replicator (t_rep): 20/20
- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 16/20 worlds; median copied 0.82 (range 0.77–0.84), median damaged 0.19 (range 0.16–0.27)
- final dominant (faithful in 20/20): control flow in 0/20; block-repeat (LDIR/LDDR) in 12/20; copies ≥ 95% of partners in 12/20 (with control flow 0/20; with control flow or block-repeat 12/20); modal final tape `11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5…` in 8/20 worlds, copied 0.73, damaged 0.26, control flow -, block -
- paired per world: gained control flow 0, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 12

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.06 |   0.09 |   0.25 |   0.31 |    0.5 |   0.44 |    0.38 |    0.31 |    0.31 |    0.31 |    0.31 |    0.31 |     0.31 |     0.34 |     0.38 |     0.44 |      0.5 |     0.44 |      0.88 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 32:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | -             |   8 |     0.80 |      0.20 |
| -          | LDIR          |  12 |     1.00 |      0.00 |
