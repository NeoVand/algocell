# Stage G — numbers (generated; do not edit)

Per-world executor tests: 256 random partners, one 128-step encounter; `copied` = fraction of partners that become a ≥ 75% copy (best cyclic shift), `damaged` = fraction of encounters in which the organism loses ≥ 25% of its bytes. Control flow = jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly (pre-registered); `block` = LDIR/LDDR-type repeat instructions (reported separately).

Rule: 10/10 worlds ran under `zero_halts` (lethal tar); their partner tests were executed on the derived lethal executor (`z80_test_lethal_*.wgsl`), the others under the normal rule.

## L = 16 (lethal@closure, horizon 300,000, 10 worlds)

- worlds with a heritable replicator (t_rep): 10/10
- first replicator: control flow in 3/10; copies < 90% of random partners in 0/10; modal first tape `11 68 4f ed b0 9b 11 dc 11 68 4f ed b0 9b 11 dc` in 1/10 worlds; median copied 1.00 (range 1.00–1.00), median damaged 0.00 (range 0.00–0.00)
- final dominant (faithful in 10/10): control flow in 3/10; block-repeat (LDIR/LDDR) in 9/10; copies ≥ 95% of partners in 10/10 (with control flow 3/10; with control flow or block-repeat 10/10); modal final tape `04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0` in 1/10 worlds, copied 1.00, damaged 0.00, control flow -, block LDIR
- paired per world: gained control flow 1, lost 1; gained closure (copied < 0.95 → ≥ 0.95) 0

heritable fraction of 16 random cells (median over worlds) by step:

|                  |   50 |   100 |   150 |   200 |   300 |   500 |   750 |   1000 |   1500 |   2000 |   3000 |   5000 |   7500 |   10000 |   15000 |   20000 |   30000 |   50000 |   75000 |   100000 |   150000 |   200000 |   300000 |
|:-----------------|-----:|------:|------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| median_heritable |    0 |     0 |     0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |      0 |      0 |       0 |       0 |       0 |    0.41 |    0.88 |    0.94 |     0.94 |     0.94 |     0.94 |     0.91 |

## Pre-registered verdicts

- not evaluated (--no-verdicts): the Stage G thresholds are pre-registered for 20 worlds per L; this directory is scored by its own pre-registration (PLAN.md).

## Final-tape classes per L

L = 16:

| final_cf      | final_block   |   n |   copied |   damaged |
|:--------------|:--------------|----:|---------:|----------:|
| -             | LDIR          |   7 |     1.00 |      0.00 |
| JP C,nn+RET Z | LDIR          |   1 |     1.00 |      0.00 |
| JP nn+JR Z,d  | -             |   1 |     1.00 |      0.00 |
| RET Z         | LDIR          |   1 |     1.00 |      0.00 |
