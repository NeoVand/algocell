# BFF soups — numbers (generated; do not edit)

Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; `t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; `t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. Culture tests: 64 random partners, 2^13 steps.

## std: 24 runs (24 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 9/24 (median 6464 epochs); heritable fraction ≥ 0.5 (t_her) in 7/24 (median 10880); pre-registered share criterion (t_rep) in 2/24; HOE ≥ 1 in 9/24 (median 10816); closed top class in 9/24; an open replicator ever in the top 3 in 3/24
- first replicators: {'closed': 7, 'intermediate': 2}; with a loop 9/9; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 7/9, with a loop 7/9; final heritable fraction median 0.97 (max over time, median 1.00, reached at median epoch 8896); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/9; first replicator a one-byte tiling in 0/9
- before t_rep: mean chunk transfer 0.47 bytes/encounter (median over runs), max 90th percentile 2, max copy-event fraction 0.0117

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     10 |  3072.00 |  3392.00 |  3200.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·<······[··,···········}·<,]··,<·}···[·····,·,··[·········<····· |
|     11 | 16064.00 | 16384.00 | 16128.00 | intermediate  | True         |            0.03 |           0.86 |                0.00 |         0.73 |         0.14 | ·<·0··[······,·}·····<·]·[··<·<··[·]·<·····}·,······[·····0·<<·> |
|     16 | 12672.00 | 12736.00 |  9024.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ··········[<,·}····]··}···,,··<·[············{····{}······{·}··} |
|     17 |  6464.00 |  6400.00 |  6336.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ····<·,·[··[·[·,·}·····<···]·-······]··<·······}·,·[[·····,·<··· |
|      2 |  3648.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ·<·······[,·<·}··,···············]···············,··}·<·,[······ |
|     20 |  2368.00 |  4992.00 |  2432.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.41 | ·····[[[[·[[[[[[[[[[[<·,}]······················]},·<[[[[[[[[[[[ |
|     22 | 11136.00 | 11200.00 | 11072.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ·[·<····}·········,]·}······,··<[···[[··················,·.····· |
|      4 | 10688.00 | 10880.00 | 10880.00 | intermediate  | True         |            0.34 |           0.72 |                0.05 |         0.56 |         0.31 | [[·[··,···<}······]·········]]·········]······}<···,··[·[[[···<· |
|      9 |  1920.00 |   nan    |   nan    | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | ····[············<,·}·]·}····,<·}·<,····}·]·}·,<············[··· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|     10 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | ·····<········,·····,·····[···}·<,··],<·}···········,··[····+,<· |
|     11 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | -·····+[[··[···[··[·············<····,··}·],····<···}···[[·,,,,< |
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·····[<,·}····]··}····,··<·[·······················}·····{····-+ |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ··00··<·,+[-·[···,·}·····<···]······-·]·<·····}·,···[··[·,·<·0·· |
|      2 |         16384 | fill          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         1.00 | ································································ |
|     20 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·,··············[·<·,}]··························},·<········[·· |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··{[·····<····}·········,]·}·····,··<·[·····+··················- |
|      4 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.08 | ·······[··········[<···,····}······]·············}····,···<[···· |
|      9 |         16384 | fill          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         1.00 | ································································ |

- runs without a replicator by the criterion: seeds [1, 3, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 21, 23, 24]; their final HOE median 0.13, final heritable fraction median 0.00

## stdlit: 12 runs (12 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 12/12 (median 64 epochs); heritable fraction ≥ 0.5 (t_her) in 5/12 (median 128); pre-registered share criterion (t_rep) in 12/12; HOE ≥ 1 in 0/12 (median nan); closed top class in 0/12; an open replicator ever in the top 3 in 12/12
- first replicators: {'open': 12}; with a loop 0/12; median entered 1.00, copies 0.91, self-damage 0.00
- final top class: closed in 0/12, with a loop 0/12; final heritable fraction median 0.00 (max over time, median 0.47, reached at median epoch 96); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 5/12; first replicator a one-byte tiling in 12/12
- before t_rep: mean chunk transfer 7.71 bytes/encounter (median over runs), max 90th percentile 46, max copy-event fraction 0.4893

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |   t_top |   t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|--------:|--------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |   64.00 |  128.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.89 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           1.00 |                0.00 |         0.91 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.95 |                0.00 |         0.92 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |   64.00 |  128.00 |      nan | open          | False        |            1.00 |           0.88 |                0.00 |         0.77 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.83 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |   64.00 |  128.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.75 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |   64.00 |  nan    |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.79 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.80 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.83 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |         16384 | open          | False        |            1.00 |           0.92 |                0.00 |         0.72 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |         16384 | open          | False        |            1.00 |           0.00 |                0.00 |         0.00 |         0.42 | 000000000000000000PPPP·,PP0PP0P·PPP·PP0P0P,·PP00P0P·PPP·,·P·PPP· |
|      4 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.73 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |         16384 | open          | False        |            1.00 |           0.97 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

## wrap: 24 runs (24 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 19/24 (median 6912 epochs); heritable fraction ≥ 0.5 (t_her) in 19/24 (median 6976); pre-registered share criterion (t_rep) in 14/24; HOE ≥ 1 in 24/24 (median 64); closed top class in 19/24; an open replicator ever in the top 3 in 2/24
- first replicators: {'closed': 16, 'intermediate': 3}; with a loop 19/19; median entered 0.00, copies 1.00, self-damage 0.00
- final top class: closed in 18/19, with a loop 18/19; final heritable fraction median 0.97 (max over time, median 1.00, reached at median epoch 7680); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/19; first replicator a one-byte tiling in 0/19
- before t_rep: mean chunk transfer 0.98 bytes/encounter (median over runs), max 90th percentile 62, max copy-event fraction 0.8066

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |    t_top |    t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|---------:|---------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |  7424.00 |  7424.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.16 | >>>>···<<,·<····<[···<·········,·,,}··],,,]··},,·,·········<···[ |
|     10 |  4224.00 |  4224.00 |    64.00 | intermediate  | True         |            0.00 |           0.89 |                0.17 |         0.81 |         0.12 | <<[[><·[·>[,··<·····}···]······--······]···}·····<··,[>·[·<>[[<< |
|     12 |  8576.00 |  8960.00 |    64.00 | intermediate  | True         |            0.00 |           0.83 |                0.00 |         0.62 |         0.22 | 0<[,····<·}··]<·<·<·<]··}·<····,,····<·}··]<·<·<·<]··}·<····,[<0 |
|     13 |  3712.00 |  3712.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ··<···[,}····<··-,]·········<··<<··<·········],-··<····},[···<·· |
|     14 |  1536.00 |  1536.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.17 | ,,,,,,,,,>>{··<<···[}<·····,·····]·····]·····,·····<}[···<<··{>> |
|     15 |  4480.00 |  4480.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.31 | 0<[[··,···}············<··,···]··,··<············}···,··[[<····0 |
|     16 |  1728.00 |  1728.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | >>>······<<<···············><<[·>····{·.·]{]·.·{····>·[<<>······ |
|     17 |  7680.00 |  7616.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.19 | {<<[[·<·}>·<,······<·>··]············]··>·<······,<·>}·<·[[<<{{{ |
|     18 |  3968.00 |  4032.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.47 | <,,,,·,,,,[,}<,····],,·,·,,·····,<},[,,,,·,,,,,·,,,·,·<<<···>··> |
|      2 | 13120.00 | 13120.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.14 | ····,[··<···,,·····}···]·····,}····},·····]···}·····,,···<··[,·· |
|     20 |   768.00 |   960.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.28 | [···{·····..····,········>···],,,]···>········,····..·····{···[· |
|     22 |  1088.00 |  1088.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.25 | <[,·,[·[[·[·[·,[[·.>{]>··.{····{{····{.··>]{>.·[[,·[·[·[[·[,·,[< |
|     23 |  7552.00 |  7552.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.39 | ··<·····<,·[,·<··}·,·]··········}··········]·,·}··<·,[·,<·····<· |
|     24 | 16320.00 | 16320.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |
|      3 | 12288.00 | 12288.00 |    64.00 | intermediate  | True         |            0.00 |           0.88 |                0.00 |         0.63 |         0.19 | <<>>><><>><<··<[,·<}],]}<·,[····[,·<}],]}<·,[<·················· |
|      6 | 13952.00 | 13888.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.19 | ·······[<,···}······]]]·]·]·]········]·]·]·]]]······}···,<[····· |
|      7 |  7680.00 |  7552.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·<0··0·,0[··········[[,·····<}]·-·]}<·····,[[··········[0,·0··0< |
|      8 |  5376.00 |  5376.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.25 | ·········,,·}<·,[·<··,··}]·,-··<·<·<··-,·]}··,··<·[,·<}·,,······ |
|      9 |  6912.00 |  6976.00 |    64.00 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·{····[··········<·····}····,]<<<<],····}·····<··········[····{· |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·········[····<···········,·}··]······},,···········<······[···· |
|     10 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·······[···<··,·····}···]·······-······]···}·····,··<·[[········ |
|     12 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.11 | ·<,··,··[···<}·,,··]··[··············]·,···]·,·}·<··[··,···<···· |
|     13 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ··[·<····,}·]·························]·},····<·[··············· |
|     14 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ··{·······[}<·····,·····]·······,·····<}[···<·····{··>········[· |
|     15 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | ···<··,·[····<}··,············]···············,··}<·[·····,·<··· |
|     16 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·······[··,·········[{·.>··]>.·{·····[·························· |
|     17 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ················{[<},············]·······,}<[{···········[······ |
|     18 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | <···········,·····[·}<·······,·····]···,·<}·[·,··+·····<<··<··>> |
|      2 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.09 | ·····[··<····,·····}···]······}···········]···}······,···<··[[·· |
|     20 |         16384 | fill          | False        |            1.00 |           0.02 |                0.09 |         0.00 |         1.00 | ................................................................ |
|     22 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | {·········[·····[·.>{·····················]{>.··[··············{ |
|     23 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ····.·[<,···}·]·····}···,<·················[·········,·········· |
|     24 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.22 | ··{····[·····<·····}·,]··]·,,··,,··,,·]··],·}·····<·····[····{·· |
|      3 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.12 | ·[····<·,[[,·<}·,]}<·,[·,·······+·······],]}<·,[[,·<············ |
|      6 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | >·<·[··················[<,···}······]]]······}···,<[············ |
|      7 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.06 | 0··<,+············[[,·····<}]·-·]}<·····,[[·······+·····,<······ |
|      8 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.03 | ···[·<··,··}]················}··,··<·[·························· |
|      9 |         16384 | closed        | True         |            0.00 |           1.00 |                0.00 |         1.00 |         0.05 | ·{····[··········<·····}····,]····],····}·····<··········[··[·{· |

- runs without a replicator by the criterion: seeds [4, 5, 11, 19, 21]; their final HOE median 0.92, final heritable fraction median 0.00

## wraplit: 12 runs (12 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 12/12 (median 64 epochs); heritable fraction ≥ 0.5 (t_her) in 12/12 (median 64); pre-registered share criterion (t_rep) in 12/12; HOE ≥ 1 in 0/12 (median nan); closed top class in 0/12; an open replicator ever in the top 3 in 12/12
- first replicators: {'open': 12}; with a loop 0/12; median entered 1.00, copies 0.91, self-damage 0.00
- final top class: closed in 0/12, with a loop 0/12; final heritable fraction median 0.00 (max over time, median 0.66, reached at median epoch 64); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 12/12; first replicator a one-byte tiling in 12/12
- before t_rep: mean chunk transfer 1.53 bytes/encounter (median over runs), max 90th percentile 64, max copy-event fraction 0.7569

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |   t_top |   t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|--------:|--------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.88 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.89 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           1.00 |                0.00 |         0.91 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.95 |                0.00 |         0.92 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.92 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.88 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.83 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.89 |                0.00 |         0.75 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |   64.00 |   64.00 |      nan | open          | False        |            1.00 |           0.91 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.80 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.82 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |         16384 | open          | False        |            1.00 |           0.92 |                0.00 |         0.72 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |         16384 | open          | False        |            1.00 |           0.92 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |         16384 | open          | False        |            1.00 |           0.89 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.70 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |         16384 | open          | False        |            1.00 |           0.97 |                0.00 |         0.91 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |         16384 | open          | False        |            1.00 |           0.86 |                0.00 |         0.78 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |         16384 | open          | False        |            1.00 |           0.94 |                0.00 |         0.85 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |         16384 | open          | False        |            1.00 |           0.95 |                0.00 |         0.86 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

## wraplitnh: 12 runs (12 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 12/12 (median 64 epochs); heritable fraction ≥ 0.5 (t_her) in 12/12 (median 64); pre-registered share criterion (t_rep) in 12/12; HOE ≥ 1 in 12/12 (median 0); closed top class in 0/12; an open replicator ever in the top 3 in 12/12
- first replicators: {'open': 12}; with a loop 0/12; median entered 1.00, copies 1.00, self-damage 0.00
- final top class: closed in 0/12, with a loop 0/12; final heritable fraction median 1.00 (max over time, median 1.00, reached at median epoch 128); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/12; first replicator a one-byte tiling in 12/12
- before t_rep: mean chunk transfer 1.35 bytes/encounter (median over runs), max 90th percentile 64, max copy-event fraction 0.9902

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |   t_top |   t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|--------:|--------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |   64.00 |   64.00 |     0.00 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     10 |         16384 | open          | False        |            1.00 |           0.97 |                0.00 |         0.94 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     11 |         16384 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|     12 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |         16384 | open          | False        |            1.00 |           0.98 |                0.00 |         0.98 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      3 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      7 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      8 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         0.97 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      9 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

## Transition rates and between-variant tests

| variant   |   runs |   transition (t_top) |   heritable ≥ 0.5 (t_her) |   median t_her (epochs; censored runs at horizon) |   HOE ≥ 1 |   HOE ≥ 1 without a replicator |   first replicator open |   first closed with loop |   final closed |   collapsed |
|:----------|-------:|---------------------:|--------------------------:|--------------------------------------------------:|----------:|-------------------------------:|------------------------:|-------------------------:|---------------:|------------:|
| std       |     24 |                    9 |                         7 |                                             16384 |         9 |                              2 |                       0 |                        7 |              7 |           0 |
| stdlit    |     12 |                   12 |                         5 |                                             16384 |         0 |                              0 |                      12 |                        0 |              0 |           5 |
| wrap      |     24 |                   19 |                        19 |                                              7552 |        24 |                              5 |                       0 |                       16 |             18 |           0 |
| wraplit   |     12 |                   12 |                        12 |                                                64 |         0 |                              0 |                      12 |                        0 |              0 |          12 |
| wraplitnh |     12 |                   12 |                        12 |                                                64 |        12 |                              0 |                      12 |                        0 |              0 |           0 |

- std vs stdlit: transitions 9/24 vs 12/12 (Fisher two-sided p = 0.000258); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 0.108
- std vs wrap: transitions 9/24 vs 19/24 (Fisher two-sided p = 0.00766); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 0.000286
- std vs wraplit: transitions 9/24 vs 12/12 (Fisher two-sided p = 0.000258); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 1.31e-07
- std vs wraplitnh: transitions 9/24 vs 12/12 (Fisher two-sided p = 0.000258); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 1.31e-07
- stdlit vs wrap: transitions 12/12 vs 19/24 (Fisher two-sided p = 0.146); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 0.824
- stdlit vs wraplit: transitions 12/12 vs 12/12 (Fisher two-sided p = 1); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 8.42e-05
- stdlit vs wraplitnh: transitions 12/12 vs 12/12 (Fisher two-sided p = 1); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 8.42e-05
- wrap vs wraplit: transitions 19/24 vs 12/12 (Fisher two-sided p = 0.146); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 8.19e-07
- wrap vs wraplitnh: transitions 19/24 vs 12/12 (Fisher two-sided p = 0.146); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 8.19e-07
- wraplit vs wraplitnh: transitions 12/12 vs 12/12 (Fisher two-sided p = 1); t_her with censored runs at the horizon, Mann–Whitney two-sided p = 1

## Readings of THEORY.md P1 (computed, pre-stated thresholds)

- (a) standard BFF: first replicators closed 7/9, with a loop 9/9, open 0/9 → NOT as predicted
- (b) wrap BFF: first replicators open 0/19, closed with a loop 16/19 → (b2) born closed
- (e1) wrap + literal: first replicators straight-line and open 12/12 transitions of 12 runs (≥ 9/12 predicted); closed with a loop 0/12 (≥ 6/12 kills) → (e1) met; (e2) earlier emergence than standard BFF: median t_her 64 vs 16384 (one-sided Mann–Whitney p = 6.56e-08) → met; (e3) final dominant closed in 0/12 transitioned worlds (≥ 6 predicted) → not met; collapsed after the open wave in 12/12 (peak heritable fraction median 0.66 at median epoch 64, final median 0.00)
- (f1) wrap + literal + no-halt: first replicators straight-line and open 12/12 transitions of 12 runs (≥ 9/12 predicted) → met; (f2) persistence: heritable fraction ≥ 0.5 at the final sample in 12/12 (≥ 9/12 predicted; collapsed 0/12, kill if ≥ 9/12) → met, against 0/12 persisting in wraplit (Fisher two-sided p = 7.4e-07); (f3) closed class dominant in 0/12 (0 predicted) → met; HOE ≥ 1 at the first sample, before any class holds 1% of the soup, in 12/12, and HOE < 0.1 at every sample from epoch 64 on in 12/12 (the detector fires before the replicator exists and is silent during its reign)
- (l1) literal without wrap: first replicators straight-line and open 12/12 transitions of 12 runs (≥ 9/12 predicted) → met; (l2) collapse by the pre-registered letter (heritable fraction ≥ 0.5, then < 0.1) 5/12 (≥ 9/12 predicted) → not met by the letter; the wave reached 0.5 in 5/12 worlds (peak heritable fraction median 0.47, range 0.38–0.62, first-replicator copies median 0.91); final heritable fraction < 0.1 in 12/12 finished worlds (no persistent open population); closed class dominant in 0/12