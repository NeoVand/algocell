# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (none@closure10M, horizon 10,000,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 0/10; copies < 90% of random partners in 10/10; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 7/10 worlds; median copied 0.65 (range 0.63–0.72), median damaged 0.34 (range 0.30–0.38)
- final dominant (faithful in 10/10): control flow in 9/10; block-repeat (LDIR/LDDR) in 1/10; copies ≥ 95% of partners in 10/10 (with control flow 9/10; with control flow or block-repeat 10/10); modal final tape `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` in 9/10 worlds, copied 1.00, damaged 0.00, control flow RET NZ, block -
- paired per world: gained control flow 9, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 10

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |  0.03 |     0 |      0 |   0.06 |   0.09 |   0.09 |   0.06 |    0.22 |    0.25 |    0.44 |    0.78 |    0.81 |    0.84 |     0.88 |     0.91 |     0.91 |     0.84 |     0.81 |     0.88 |      0.88 |

## L = 20 (none@closure10M, horizon 10,000,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 0/10; copies < 90% of random partners in 10/10; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 9/10 worlds; median copied 0.67 (range 0.63–0.69), median damaged 0.43 (range 0.41–0.49)
- final dominant (faithful in 10/10): control flow in 1/10; block-repeat (LDIR/LDDR) in 9/10; copies ≥ 95% of partners in 10/10 (with control flow 1/10; with control flow or block-repeat 10/10); modal final tape `1e 7c ed b0 1e 7c ed b0 1e 7c ed b0 1e 7c ed b0…` in 3/10 worlds, copied 1.00, damaged 0.00, control flow -, block LDIR
- paired per world: gained control flow 1, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 10

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |   0.03 |   0.06 |   0.06 |    0.06 |    0.06 |    0.06 |    0.09 |    0.19 |    0.12 |     0.12 |     0.22 |     0.53 |     0.88 |     0.94 |     0.94 |      0.94 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 16:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | LDIR          |   1 |     1.00 |      0.00 |
| RET NZ     | -             |   9 |     1.00 |      0.00 |

L = 20:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| -          | LDDR          |   1 |     1.00 |      1.00 |
| -          | LDIR          |   8 |     1.00 |      0.00 |
| JR NC,d    | -             |   1 |     1.00 |      0.00 |
