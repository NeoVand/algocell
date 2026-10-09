# BFF soups — numbers (generated; do not edit)

Events: `t_top` = first sample at which the most common tape class is heritable (gen2 ≥ 0.3) and not a one-symbol fill (one byte ≥ 90% of the tape; fills are tar, reported as class 'fill'); it is the exemplar used as the first replicator; `t_her` = heritable fraction of 32 random tapes ≥ 0.5; `t_rep` = the pre-registered Z80 criterion (top-3 class share ≥ 0.5% and gen2 ≥ 0.3), kept for the record; `t_hoe1` = high-order entropy ≥ 1 bit/byte; `t_closed` = top class enters the partner ≤ 5% and copies ≥ 95% of random partners; `t_open` = a replicating top-3 class enters the partner ≥ 50%. Culture tests: 64 random partners, 2^13 steps.

## stdlit: 6 runs (6 finished; epochs done median 16,384)

- transitions: top class heritable (t_top) in 6/6 (median 64 epochs); heritable fraction ≥ 0.5 (t_her) in 6/6 (median 64); pre-registered share criterion (t_rep) in 6/6; HOE ≥ 1 in 6/6 (median 64); closed top class in 0/6; an open replicator ever in the top 3 in 6/6
- first replicators: {'open': 6}; with a loop 0/6; median entered 1.00, copies 1.00, self-damage 0.00
- final top class: closed in 0/6, with a loop 0/6; final heritable fraction median 1.00 (max over time, median 1.00, reached at median epoch 64); collapsed (heritable fraction ≥ 0.5 reached, < 0.1 at the end) 0/6; first replicator a one-byte tiling in 5/6
- before t_rep: mean chunk transfer 26.21 bytes/encounter (median over runs), max 90th percentile 64, max copy-event fraction 1.0000

first replicators (BFF string; `·` = non-instruction byte, `0` = zero):

|   seed |   t_top |   t_her |   t_hoe1 | first_class   | first_loop   |   first_entered |   first_copies |   first_self_damage |   first_gen2 |   first_fill | first_pretty                                                     |
|-------:|--------:|--------:|---------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |      64 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |       0 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         0.50 | P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P· |
|      3 |      64 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |      64 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      5 |      64 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      6 |      64 |      64 |       64 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |

final top classes:

|   seed |   final_epoch | final_class   | final_loop   |   final_entered |   final_copies |   final_self_damage |   final_gen2 |   final_fill | final_pretty                                                     |
|-------:|--------------:|:--------------|:-------------|----------------:|---------------:|--------------------:|-------------:|-------------:|:-----------------------------------------------------------------|
|      1 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      2 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         0.50 | P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P· |
|      3 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         1.00 | PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP |
|      4 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         0.50 | P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P· |
|      5 |         16384 | open          | False        |            1.00 |           1.00 |                1.00 |         1.00 |         0.50 | ·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P |
|      6 |         16384 | open          | False        |            1.00 |           1.00 |                0.00 |         1.00 |         0.50 | P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P·P· |

## Transition rates and between-variant tests

| variant   |   runs |   transition (t_top) |   heritable ≥ 0.5 (t_her) |   median t_her (epochs; censored runs at horizon) |   HOE ≥ 1 |   HOE ≥ 1 without a replicator |   first replicator open |   first closed with loop |   final closed |   collapsed |
|:----------|-------:|---------------------:|--------------------------:|--------------------------------------------------:|----------:|-------------------------------:|------------------------:|-------------------------:|---------------:|------------:|
| stdlit    |      6 |                    6 |                         6 |                                                64 |         6 |                              0 |                       6 |                        0 |              0 |           0 |


## Readings of THEORY.md P1 (computed, pre-stated thresholds)

- (l1) literal without wrap: first replicators straight-line and open 6/6 transitions of 6 runs (≥ 9/12 predicted) → not met; (l2) collapse by the pre-registered letter (heritable fraction ≥ 0.5, then < 0.1) 0/6 (≥ 9/12 predicted) → not met by the letter; the wave reached 0.5 in 6/6 worlds (peak heritable fraction median 1.00, range 1.00–1.00, first-replicator copies median 1.00); final heritable fraction < 0.1 in 0/6 finished worlds (no persistent open population); closed class dominant in 0/6