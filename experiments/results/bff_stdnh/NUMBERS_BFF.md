# BFF soups — numbers (generated; do not edit)

Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; `t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; `t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. Culture tests: 64 random partners, 2^13 steps.

## stdnh: 12 runs (12 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 7/12 (median 11200 epochs); heritable fraction ≥ 0.5 (t_her) in 3/12 (median 11328); pre-registered share criterion (t_rep) in 0/12; HOE ≥ 1 in 3/12 (median 11328); closed top class in 6/12; an open replicator ever in the top 3 in 2/12
- first replicators: {'closed': 5, 'open': 1, 'intermediate': 1}; with a loop 7/7; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 3/7, with a loop 6/7; final heritable fraction median 0.00 (max over time, median 0.00, reached at median epoch 0); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/7; first replicator a one-byte tiling in 0/7
- before t_rep: mean chunk transfer 0.38 bytes/encounter (median over runs), max 90th percentile 1, max copy-event fraction 0.0020

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     11 |  5568.00 |  5696.00 |  5504.00 | open          | True         |            1.00 |           0.81 |                0.00 |         0.71 |         0.16 | ·000<·0····[·,·<··}··············]}·····<···,·00[··0·[0[·0·0···· |
|     12 | 15808.00 | 16256.00 | 16128.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | <·[[[[[·······,<}··,,···,·,··]]··,·,···,,··}<,·······[[[[[······ |
|      2 | 11200.00 | 11328.00 | 11328.00 | closed        | True         |            0.03 |           0.97 |                0.00 |         0.94 |         0.17 | ·<·[}··<,··]]··,<··}[·<····<················>········<<<<,·<···< |
|      4 |  3136.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | [<·,··}·······]···················<]···············}·····,···<·[ |
|      5 |  3008.00 |   nan    |   nan    | intermediate  | True         |            0.23 |           0.84 |                0.08 |         0.61 |         0.14 | ········>··<·····]···<·····}·,··[·············,}··<·]··<··.····· |
|      7 | 14400.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ·············<·····<··]}.····,··<····[·······[····<··,····.}]··· |
|      9 | 12032.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | ·<···········[··{······.·>]·······]··]·······]>·.······{··[····· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     11 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ,·]···{····[<},················]···}····,··<·············[······ |
|     12 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | ·····<]···[·······,<}··,,···,·,··]]··,·,···,,··}<,·······[·····< |
|      2 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | >···········<··{[·}·····<·,·····]··,····<·····}·[········]····{· |
|      4 |         16384 | fill          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         1.00 | [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ |
|      5 |         16384 | open          | True         |            1.00 |           0.00 |                0.00 |         0.00 |         0.42 | 0000000000000000000000···<·············]···+[·····,·····<·>00000 |
|      7 |         16384 | intermediate  | True         |            0.00 |           0.00 |                0.00 |         0.00 |         0.25 | 000000000000·<00][··][··]·[··]+[·+0·,···················<····,0, |
|      9 |         16384 | open          | True         |            1.00 |           0.00 |                0.00 |         0.00 |         0.30 | 000000000000·0·············}···]···+··[······0········0··00···00 |

- runs without a replicator by the criterion: seeds [1, 3, 6, 8, 10]; their final HOE median 0.04, final heritable fraction median 0.00

## Transition rates and between-variant tests

| variant   |   runs |   transition (t_top) |   heritable ≥ 0.5 (t_her) |   median t_her (epochs; censored runs at horizon) |   HOE ≥ 1 |   HOE ≥ 1 without a replicator |   first replicator open |   first closed with loop |   final closed |   collapsed |
|:----------|-------:|---------------------:|--------------------------:|--------------------------------------------------:|----------:|-------------------------------:|------------------------:|-------------------------:|---------------:|------------:|
| stdnh     |     12 |                    7 |                         3 |                                             16384 |         3 |                              0 |                       1 |                        5 |              3 |           0 |


## Readings of THEORY.md P1 (computed, pre-stated thresholds)
