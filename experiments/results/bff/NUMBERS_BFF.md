# BFF soups — numbers (generated; do not edit)

Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; `t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; `t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. Culture tests: 64 random partners, 2^13 steps.

## std: 15 runs (15 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 4/15 (median 8800 epochs); heritable fraction ≥ 0.5 (t_her) in 4/15 (median 8800); pre-registered share criterion (t_rep) in 0/15; HOE ≥ 1 in 6/15 (median 9920); closed top class in 4/15; an open replicator ever in the top 3 in 1/15
- first replicators: {'closed': 4}; with a loop 4/4; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 4/4, with a loop 4/4; final heritable fraction median 0.98 (max over time, median 1.00)
- before t_rep: mean chunk transfer 0.48 bytes/encounter (median over runs), max 90th percentile 61, max copy-event fraction 0.3271

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     16 | 12672.00 | 12736.00 |  9024.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ··········[<,·}····]··}···,,··<·[············{····{}······{·}··} |
|     17 |  6464.00 |  6400.00 |  6336.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ····<·,·[··[·[·,·}·····<···]·-······]··<·······}·,·[[·····,·<··· |
|     20 |  2368.00 |  4992.00 |  2432.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.41 | ·····[[[[·[[[[[[[[[[[<·,}]······················]},·<[[[[[[[[[[[ |
|     22 | 11136.00 | 11200.00 | 11072.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ·[·<····}·········,]·}······,··<[···[[··················,·.····· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·····[<,·}····]··}····,··<·[·······················}·····{····-+ |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ··00··<·,+[-·[···,·}·····<···]······-·]·<·····}·,···[··[·,·<·0·· |
|     20 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·,··············[·<·,}]··························},·<········[·· |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··{[·····<····}·········,]·}·····,··<·[·····+··················- |

- runs without a replicator by the criterion: seeds [1, 6, 7, 13, 14, 15, 18, 19, 21, 23, 24]; their final HOE median 0.13, final heritable fraction median 0.00

## wrap: 13 runs (13 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 11/13 (median 3968 epochs); heritable fraction ≥ 0.5 (t_her) in 11/13 (median 4032); pre-registered share criterion (t_rep) in 8/13; HOE ≥ 1 in 13/13 (median 64); closed top class in 11/13; an open replicator ever in the top 3 in 2/13
- first replicators: {'closed': 10, 'intermediate': 1}; with a loop 11/11; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 10/11, with a loop 10/11; final heritable fraction median 0.97 (max over time, median 1.00)
- before t_rep: mean chunk transfer 1.26 bytes/encounter (median over runs), max 90th percentile 61, max copy-event fraction 0.8184

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     12 |  8576.00 |  8960.00 |    64.00 | intermediate  | True         |            0.00 |           0.83 |                0.00 |         0.62 |         0.22 | 0<[,····<·}··]<·<·<·<]··}·<····,,····<·}··]<·<·<·<]··}·<····,[<0 |
|     13 |  3712.00 |  3712.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ··<···[,}····<··-,]·········<··<<··<·········],-··<····},[···<·· |
|     14 |  1536.00 |  1536.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.17 | ,,,,,,,,,>>{··<<···[}<·····,·····]·····]·····,·····<}[···<<··{>> |
|     15 |  4480.00 |  4480.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.31 | 0<[[··,···}············<··,···]··,··<············}···,··[[<····0 |
|     16 |  1728.00 |  1728.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | >>>······<<<···············><<[·>····{·.·]{]·.·{····>·[<<>······ |
|     17 |  7680.00 |  7616.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.19 | {<<[[·<·}>·<,······<·>··]············]··>·<······,<·>}·<·[[<<{{{ |
|     18 |  3968.00 |  4032.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.47 | <,,,,·,,,,[,}<,····],,·,·,,·····,<},[,,,,·,,,,,·,,,·,·<<<···>··> |
|     20 |   768.00 |   960.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.28 | [···{·····..····,········>···],,,]···>········,····..·····{···[· |
|     22 |  1088.00 |  1088.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.25 | <[,·,[·[[·[·[·,[[·.>{]>··.{····{{····{.··>]{>.·[[,·[·[·[[·[,·,[< |
|     23 |  7552.00 |  7552.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.39 | ··<·····<,·[,·<··}·,·]··········}··········]·,·}··<·,[·,<·····<· |
|     24 | 16320.00 | 16320.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     12 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | ·<,··,··[···<}·,,··]··[··············]·,···]·,·}·<··[··,···<···· |
|     13 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··[·<····,}·]·························]·},····<·[··············· |
|     14 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ··{·······[}<·····,·····]·······,·····<}[···<·····{··>········[· |
|     15 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ···<··,·[····<}··,············]···············,··}<·[·····,·<··· |
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·······[··,·········[{·.>··]>.·{·····[·························· |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ················{[<},············]·······,}<[{···········[······ |
|     18 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | <···········,·····[·}<·······,·····]···,·<}·[·,··+·····<<··<··>> |
|     20 |         16384 | fill          | False        |            1.00 |           0.02 |                0.09 |         0.00 |         1.00 | ................................................................ |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | {·········[·····[·.>{·····················]{>.··[··············{ |
|     23 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ····.·[<,···}·]·····}···,<·················[·········,·········· |
|     24 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |

- runs without a replicator by the criterion: seeds [19, 21]; their final HOE median 0.93, final heritable fraction median 0.00

## Transition rates and between-variant tests

| variant   |   runs |   transition (t_top) |   heritable ≥ 0.5 (t_her) |   median t_her (epochs; censored runs at horizon) |   HOE ≥ 1 |   HOE ≥ 1 without a replicator |   first replicator open |   first closed with loop |   final closed |
|:----------|-------:|---------------------:|--------------------------:|--------------------------------------------------:|----------:|-------------------------------:|------------------------:|-------------------------:|---------------:|
| std       |     15 |                    4 |                         4 |                                             16384 |         6 |                              2 |                       0 |                        4 |              4 |
| wrap      |     13 |                   11 |                        11 |                                              4480 |        13 |                              2 |                       0 |                       10 |             10 |

- std vs wrap: transitions 4/15 vs 11/13 (Fisher two-sided p = 0.00323); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 0.000883

## Readings of THEORY.md P1 (computed, pre-stated thresholds)

- (a) standard BFF: first replicators closed 4/4, with a loop 4/4, open 0/4 → confirmed
- (b) wrap BFF: first replicators open 0/11, closed with a loop 10/11 → (b2) born closed