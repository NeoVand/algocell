# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (i8080@closure, horizon 300,000, 20 worlds)

- worlds with a heritable replicator (t_rep): 20/20
- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 13/20 worlds; median copied 0.67 (range 0.59–0.75), median damaged 0.33 (range 0.27–0.38)
- final dominant (faithful in 20/20): control flow in 20/20; block-repeat (LDIR/LDDR) in 0/20; copies ≥ 95% of partners in 20/20 (with control flow 20/20; with control flow or block-repeat 20/20); modal final tape `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` in 20/20 worlds, copied 1.00, damaged 0.00, control flow RET NZ, block -
- paired per world: gained control flow 20, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 20

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.06 |   0.03 |   0.06 |   0.09 |   0.12 |   0.06 |   0.12 |    0.16 |    0.12 |    0.22 |    0.44 |    0.81 |    0.81 |     0.88 |     0.81 |     0.88 |     0.88 |

## L = 32 (i8080@closure1M, horizon 1,000,000, 20 worlds)

- worlds with a heritable replicator (t_rep): 20/20
- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 14/20 worlds; median copied 0.81 (range 0.74–0.86), median damaged 0.19 (range 0.15–0.30)
- final dominant (faithful in 20/20): control flow in 0/20; block-repeat (LDIR/LDDR) in 0/20; copies ≥ 95% of partners in 0/20 (with control flow 0/20; with control flow or block-repeat 0/20); modal final tape `11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5…` in 12/20 worlds, copied 0.80, damaged 0.19, control flow -, block -
- paired per world: gained control flow 0, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 0

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.06 |   0.19 |   0.34 |   0.38 |   0.53 |    0.5 |    0.38 |    0.44 |    0.38 |    0.38 |    0.38 |    0.41 |     0.44 |     0.41 |     0.47 |     0.44 |     0.41 |     0.56 |       0.5 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 16:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| RET NZ     | -             |  20 |     1.00 |      0.00 |

L = 32:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | -             |  20 |     0.80 |      0.19 |
