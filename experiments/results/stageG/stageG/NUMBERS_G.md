# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (none@closure, horizon 300,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 19/20 worlds; median copied 0.68 (range 0.62–0.75), median damaged 0.33 (range 0.27–0.43)
- final dominant (faithful in 20/20): control flow in 20/20; block-repeat (LDIR/LDDR) in 2/20; copies ≥ 95% of partners in 20/20 (with control flow 20/20; with control flow or block-repeat 20/20); modal final tape `ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0` in 17/20 worlds, copied 1.00, damaged 0.00, control flow RET NZ, block -
- paired per world: gained control flow 20, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 20

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |   0.03 |   0.06 |   0.12 |   0.12 |   0.12 |   0.09 |    0.12 |    0.19 |    0.19 |    0.75 |    0.88 |    0.81 |     0.88 |     0.88 |     0.94 |     0.91 |

## L = 20 (none@closure1M, horizon 1,000,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 12/20 worlds; median copied 0.66 (range 0.56–0.71), median damaged 0.43 (range 0.39–0.48)
- final dominant (faithful in 20/20): control flow in 6/20; block-repeat (LDIR/LDDR) in 15/20; copies ≥ 95% of partners in 19/20 (with control flow 6/20; with control flow or block-repeat 19/20); modal final tape `1e 04 ed b0 1e 04 ed b0 1e 04 ed b0 1e 04 ed b0…` in 4/20 worlds, copied 1.00, damaged 0.00, control flow -, block LDIR
- paired per world: gained control flow 6, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 19

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |   0.06 |   0.12 |    0.12 |    0.16 |    0.12 |    0.19 |    0.19 |    0.19 |     0.16 |     0.16 |     0.28 |     0.72 |     0.88 |     0.94 |      0.94 |

## L = 50 (none@closure, horizon 300,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 14/20 worlds; median copied 0.68 (range 0.59–0.78), median damaged 0.11 (range 0.08–0.17)
- final dominant (faithful in 20/20): control flow in 20/20; block-repeat (LDIR/LDDR) in 0/20; copies ≥ 95% of partners in 20/20 (with control flow 20/20; with control flow or block-repeat 20/20); modal final tape `01 c5 01 c5 01 c5 01 c5 01 20 f0 c5 01 c5 01 c5…` in 14/20 worlds, copied 1.00, damaged 0.00, control flow JR NZ,d, block -
- paired per world: gained control flow 20, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 20

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.06 |   0.12 |   0.38 |   0.59 |    0.5 |   0.41 |   0.44 |     0.5 |     0.5 |     0.5 |     0.5 |    0.56 |    0.56 |     0.62 |     0.56 |     0.56 |     0.59 |

## L = 64 (none@closure1M, horizon 1,000,000, 20 worlds)

- first replicator: control flow in 0/20; copies < 90% of random partners in 20/20; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 18/20 worlds; median copied 0.73 (range 0.68–0.84), median damaged 0.06 (range 0.04–0.10)
- final dominant (faithful in 20/20): control flow in 3/20; block-repeat (LDIR/LDDR) in 7/20; copies ≥ 95% of partners in 8/20 (with control flow 3/20; with control flow or block-repeat 8/20); modal final tape `11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5…` in 5/20 worlds, copied 0.79, damaged 0.09, control flow -, block -
- paired per world: gained control flow 3, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 8

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |   500000 |   750000 |   1000000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |  0.03 |   0.06 |   0.25 |   0.38 |   0.44 |   0.38 |    0.25 |    0.22 |    0.31 |    0.25 |    0.28 |    0.19 |     0.25 |     0.25 |     0.25 |     0.25 |     0.31 |     0.25 |      0.34 |

## Pre-registered verdicts

- **G1, L = 16**: (a) first without control flow 20/20 (≥ 18 needed) and partner-dependent 20/20 (≥ 18): met; (b) control-flow faithful dominant 20/20 (≥ 16), modal final copied 1.00 (≥ 0.95), damaged 0.00 (≤ 0.05): met; (c) median heritable fraction ≤ 5k after emergence all < 0.20, at 100k 0.88 (> 0.70): met; kill criterion not triggered. Behavioural closure (copied ≥ 0.95) in 20/20 finals; with a loop of either kind 20/20.
- **G2, L = 20**: faithful control-flow dominant at 1M copying ≥ 95% of partners in 6/20 (prediction ≥ 10, alternative < 5): between; with a loop of either kind (control flow or LDIR/LDDR) 19/20; closed by partner test regardless of syntax 19/20; finals still the open pusher (no control flow, no block-repeat, copied < 0.95): 1/20.
- **G1, L = 50**: (a) first without control flow 20/20 (≥ 18 needed) and partner-dependent 20/20 (≥ 18): met; (b) control-flow faithful dominant 20/20 (≥ 16), modal final copied 1.00 (≥ 0.95), damaged 0.00 (≤ 0.05): met; (c) median heritable fraction ≤ 5k after emergence max 0.62, at 100k 0.62 (> 0.70): NOT met; kill criterion not triggered. Behavioural closure (copied ≥ 0.95) in 20/20 finals; with a loop of either kind 20/20.
- **G2, L = 64**: faithful control-flow dominant at 1M copying ≥ 95% of partners in 3/20 (prediction ≥ 10, alternative < 5): alternative; with a loop of either kind (control flow or LDIR/LDDR) 8/20; closed by partner test regardless of syntax 8/20; finals still the open pusher (no control flow, no block-repeat, copied < 0.95): 12/20.

## Final-tape classes per L

L = 16:

| final_cf                       | final_block   |   n |   copied |   damaged |
|:-------------------------------|:--------------|----:|---------:|----------:|
| JP (HL)+JP NC,nn+JP nn+JR NZ,d | LDIR          |   1 |     1.00 |      0.00 |
| JP nn                          | -             |   1 |     1.00 |      0.00 |
| JR Z,d                         | LDIR          |   1 |     1.00 |      0.00 |
| RET NZ                         | -             |  17 |     1.00 |      0.00 |

L = 20:

| final_cf        | final_block   |   n |   copied |   damaged |
|:----------------|:--------------|----:|---------:|----------:|
| -               | -             |   1 |     0.69 |      0.36 |
| -               | LDIR          |  13 |     1.00 |      0.00 |
| DJNZ d          | -             |   3 |     1.00 |      0.00 |
| DJNZ d+JP NZ,nn | LDDR          |   1 |     1.00 |      0.00 |
| JR NZ,d         | -             |   1 |     1.00 |      0.00 |
| RET             | LDDR          |   1 |     1.00 |      0.00 |

L = 50:

| final_cf   | final_block   |   n |   copied |   damaged |
|:-----------|:--------------|----:|---------:|----------:|
| JR NZ,d    | -             |  20 |     1.00 |      0.00 |

L = 64:

| final_cf                 | final_block   |   n |   copied |   damaged |
|:-------------------------|:--------------|----:|---------:|----------:|
| -                        | -             |  12 |     0.78 |      0.06 |
| -                        | LDIR          |   5 |     1.00 |      0.00 |
| CALL PE,nn+RST 08+RST 28 | LDDR          |   1 |     1.00 |      0.00 |
| DJNZ d+RST 08            | LDDR          |   1 |     1.00 |      0.00 |
| JP (HL)                  | -             |   1 |     1.00 |      0.00 |
