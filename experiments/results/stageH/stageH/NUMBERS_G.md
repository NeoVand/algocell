# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

## L = 16 (mixed@closure, horizon 300,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 0/10; copies < 90% of random partners in 10/10; modal first tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 9/10 worlds; median copied 0.66 (range 0.64–0.71), median damaged 0.33 (range 0.27–0.41)
- final dominant (faithful in 10/10): control flow in 5/10; block-repeat (LDIR/LDDR) in 1/10; copies ≥ 95% of partners in 5/10 (with control flow 5/10; with control flow or block-repeat 5/10); modal final tape `01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5` in 5/10 worlds, copied 0.69, damaged 0.30, control flow -, block -
- paired per world: gained control flow 5, lost 0; gained closure (copied < 0.95 → ≥ 0.95) 5

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |  0.03 |  0.16 |  0.16 |  0.16 |  0.12 |   0.16 |   0.09 |   0.09 |   0.12 |   0.16 |   0.25 |    0.25 |    0.38 |    0.22 |     0.5 |    0.78 |    0.72 |     0.75 |     0.75 |     0.72 |     0.72 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 16:

| final_cf        | final_block   |   n |   copied |   damaged |
|:----------------|:--------------|----:|---------:|----------:|
| -               | -             |   5 |     0.68 |      0.30 |
| JP NZ,nn+RET NC | LDIR          |   1 |     1.00 |      0.00 |
| RET NZ          | -             |   1 |     1.00 |      0.00 |
| RET PO          | -             |   3 |     1.00 |      0.00 |
