# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (none@closure10M, horizon 10,000,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 0/10; copies < 90% of random partners in 10/10; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 8/10 worlds; median copied 0.66 (range 0.63–0.72), median damaged 0.34 (range 0.27–0.38)
- final dominant (faithful in 10/10): control flow in 8/10; block-repeat (LDIR/LDDR) in 2/10; copies ≥ 95% of partners in 10/10 (with control flow 8/10; with control flow or block-repeat 10/10); modal final tape `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` in 8/10 worlds, copied 1.00, damaged 0.00, control flow RET NZ, block -
- paired per world: gained control flow 8, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 10

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |   0.06 |   0.03 |   0.16 |   0.12 |   0.12 |    0.12 |    0.16 |    0.16 |    0.69 |    0.84 |    0.81 |     0.88 |     0.91 |     0.88 |     0.81 |     0.84 |     0.94 |      0.88 |

## L = 20 (none@closure10M, horizon 10,000,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 0/10; copies < 90% of random partners in 10/10; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 7/10 worlds; median copied 0.65 (range 0.62–0.69), median damaged 0.43 (range 0.41–0.49)
- final dominant (faithful in 10/10): control flow in 0/10; block-repeat (LDIR/LDDR) in 10/10; copies ≥ 95% of partners in 10/10 (with control flow 0/10; with control flow or block-repeat 10/10); modal final tape `1e 2c ed b0 1e 2c ed b0 1e 2c ed b0 1e 2c ed b0…` in 4/10 worlds, copied 1.00, damaged 0.00, control flow -, block LDIR
- paired per world: gained control flow 0, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 10

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |   0.06 |      0 |      0 |    0.12 |    0.06 |    0.12 |    0.12 |    0.06 |    0.12 |     0.16 |     0.22 |     0.69 |     0.97 |     0.91 |     0.94 |      0.97 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 16:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | LDIR          |   2 |     1.00 |      0.00 |
| RET NZ     | -             |   8 |     1.00 |      0.00 |

L = 20:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | LDDR          |   2 |     1.00 |      1.00 |
| -          | LDIR          |   8 |     1.00 |      0.00 |
