# Closure (generated)

## 1. Control flow in the first replicator vs the final faithful dominant (`none`, 128 steps, 1/16; Stages stageB,stageC,stageE)

- first replicators with any control-flow instruction: 18 / 256
- final faithful dominants with any control-flow instruction: 105 / 233 (runs with both tapes)
- paired: gained control flow 92, lost 3, both 13, neither 125; exact McNemar two-sided p = 7.2e-24

|   L |   n |   first_cf |   final_cf | final_ops                                  |
|----:|----:|-----------:|-----------:|:-------------------------------------------|
|   5 |   9 |          8 |          9 | JP nn:6, JP NZ,nn:1, JP NC,nn:1            |
|   6 |   2 |          0 |          2 | JP NC,nn:1, JP PO,nn:1                     |
|   8 |  10 |          0 |          8 | RET P:3, JP nn+JR Z,d:1, JP nn+RET PO:1    |
|   9 |   8 |          3 |          7 | JP nn:2, JP M,nn:1, JP C,nn+JP nn:1        |
|  10 |   3 |          1 |          2 | JP nn:1, JP NC,nn:1                        |
|  12 |   6 |          3 |          4 | JP PO,nn:1, JP nn:1, JP NZ,nn:1            |
|  16 |  20 |          0 |         20 | RET NZ:18, JR d+RET NZ:1, JP nn+RET Z:1    |
|  18 |  10 |          1 |          1 | JP nn+JR Z,d:1                             |
|  20 |  10 |          0 |          0 |                                            |
|  24 |  10 |          0 |          8 | JR d:6, CALL NC,nn+JR d:1, JR NZ,d:1       |
|  25 |  20 |          0 |          0 |                                            |
|  32 |  10 |          0 |          1 | JR C,d:1                                   |
|  36 |  30 |          0 |         14 | DJNZ d:13, JR d:1                          |
|  49 |  19 |          0 |          0 |                                            |
|  50 |  10 |          0 |         10 | JR NZ,d:10                                 |
|  64 |  20 |          0 |          0 |                                            |
|  81 |  18 |          0 |          2 | RET C+RET NZ+RST 38:1, CALL PE,nn+RST 38:1 |
| 100 |  18 |          0 |         17 | JP NZ,nn:17                                |

## 2–3. The modal first and final tapes per L: convergence, control flow and partner-independence (256 random partners, one 128-step encounter)

|   L | which   |   n_tapes |   seeds_identical |   distinct |   period | control_flow   |   partners_copied |   self_damaged |   gen2_isolated | tape                                                                                                        |
|----:|:--------|----------:|------------------:|-----------:|---------:|:---------------|------------------:|---------------:|----------------:|:------------------------------------------------------------------------------------------------------------|
|   5 | first   |         9 |                 1 |          9 |        5 | -              |              1.00 |           0.00 |            1.00 | 1d 63 ed b0 1e                                                                                              |
|   5 | final   |         9 |                 5 |          5 |        5 | JP nn          |              1.00 |           0.00 |            1.00 | 00 1d c3 ed b0                                                                                              |
|   6 | first   |         9 |                 2 |          8 |        4 | -              |              0.00 |           0.00 |            1.00 | b0 00 14 ed b0 00                                                                                           |
|   6 | final   |         2 |                 1 |          2 |        6 | JP PO,nn       |              1.00 |           0.00 |            1.00 | 1e a2 e2 1c ed b0                                                                                           |
|   8 | first   |        10 |                 8 |          3 |        2 | -              |              0.58 |           0.39 |            0.53 | 01 c5 01 c5 01 c5 01 c5                                                                                     |
|   8 | final   |        10 |                 3 |          8 |        8 | RET P          |              1.00 |           0.00 |            1.00 | bc e3 21 e3 21 f0 bc f0                                                                                     |
|   9 | first   |         8 |                 3 |          6 |        3 | -              |              1.00 |           0.00 |            1.00 | 1d ed b0 1d ed b0 1d ed b0                                                                                  |
|   9 | final   |         8 |                 1 |          8 |        9 | JP (HL)        |              1.00 |           0.00 |            1.00 | 00 48 1e 99 70 ed b0 57 e9                                                                                  |
|  10 | first   |         4 |                 1 |          4 |        2 | -              |              0.55 |           0.40 |            0.31 | 01 c5 01 c5 01 c5 01 c5 01 c5                                                                               |
|  10 | final   |         3 |                 1 |          3 |        5 | -              |              1.00 |           0.00 |            1.00 | 05 6e eb ed b0 05 6e eb ed b0                                                                               |
|  12 | first   |         6 |                 1 |          6 |        4 | -              |              1.00 |           0.00 |            1.00 | 14 14 ed b0 14 14 ed b0 14 14 ed b0                                                                         |
|  12 | final   |         6 |                 1 |          6 |       12 | JR d           |              1.00 |           0.00 |            1.00 | 1e 3c 18 f1 9c ed b0 78 00 0a b3 18                                                                         |
|  16 | first   |        20 |                11 |          3 |        2 | -              |              0.64 |           0.38 |            0.59 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                             |
|  16 | final   |        20 |                18 |          3 |        8 | RET NZ         |              1.00 |           0.00 |            1.00 | ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0                                                             |
|  18 | first   |        10 |                 4 |          4 |        2 | -              |              0.54 |           0.37 |            0.43 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                       |
|  18 | final   |        10 |                 3 |          5 |        4 | -              |              1.00 |           0.00 |            1.00 | b0 62 14 ed b0 62 14 ed b0 62 14 ed b0 62 14 ed b0 62                                                       |
|  20 | first   |        10 |                 6 |          2 |        2 | -              |              0.67 |           0.46 |            0.46 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                 |
|  20 | final   |        10 |                 8 |          3 |        2 | -              |              0.63 |           0.44 |            0.47 | 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5 21 e5                                                 |
|  24 | first   |        10 |                 8 |          2 |        2 | -              |              0.77 |           0.22 |            0.60 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                     |
|  24 | final   |        10 |                 6 |          5 |        6 | JR d           |              1.00 |           1.00 |            1.00 | 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5 01 c5 01 18 f8 c5                                     |
|  25 | first   |        20 |                13 |          4 |        2 | -              |              0.79 |           0.89 |            0.70 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                  |
|  25 | final   |        20 |                 9 |          5 |        2 | -              |              0.84 |           0.90 |            0.70 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                  |
|  32 | first   |        10 |                 7 |          2 |        2 | -              |              0.80 |           0.21 |            0.72 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5             |
|  32 | final   |        10 |                 7 |          3 |        2 | -              |              0.75 |           0.27 |            0.66 | 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5             |
|  36 | first   |        30 |                20 |          2 |        2 | -              |              0.77 |           0.18 |            0.64 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |
|  36 | final   |        30 |                14 |          5 |        2 | -              |              0.72 |           0.16 |            0.64 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |
|  49 | first   |        20 |                18 |          3 |        2 | -              |              0.79 |           0.15 |            0.58 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  49 | final   |        19 |                 6 |          6 |        2 | -              |              0.68 |           0.77 |            0.62 | d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 …                                                           |
|  50 | first   |        10 |                 9 |          2 |        2 | -              |              0.67 |           0.10 |            0.58 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  50 | final   |        10 |                 7 |          2 |       14 | JR NZ,d        |              1.00 |           0.00 |            0.88 | 01 c5 01 c5 01 c5 01 c5 01 20 f0 c5 01 c5 01 c5 …                                                           |
|  64 | first   |        20 |                14 |          3 |        2 | -              |              0.72 |           0.06 |            0.62 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
|  64 | final   |        20 |                12 |          3 |        2 | -              |              0.79 |           0.05 |            0.73 | 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 d5 …                                                           |
|  81 | first   |        20 |                19 |          2 |        2 | -              |              0.79 |           0.67 |            0.68 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 …                                                           |
|  81 | final   |        18 |                12 |          4 |        2 | -              |              0.86 |           0.70 |            0.68 | c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 …                                                           |
| 100 | first   |        30 |                23 |          7 |        2 | -              |              0.82 |           0.02 |            0.72 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                                           |
| 100 | final   |        18 |                17 |          2 |       10 | JP NZ,nn       |              1.00 |           0.00 |            0.99 | c2 c5 01 c5 01 c5 01 c5 c2 c5 c2 c5 01 c5 01 c5 …                                                           |
