# Numbers for the findings (generated; do not edit)

## 1. Emergence by tape length (mutation 1/16)

### lethal@closure

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     16 |                  0 | 10/10             | 0/10            | 10/10 (38,000)  |

## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)


## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0

|                             |   n |   zero_final_mean |   zero_final_min |   zero_final_max |   zero_emergence_mean |   H0_final_mean |   steps_run_min |
|:----------------------------|----:|------------------:|-----------------:|-----------------:|----------------------:|----------------:|----------------:|
| ('lethal@closure', 16, 128) |  10 |             0.038 |            0.011 |            0.046 |                   nan |           6.802 |          300000 |

## 4. Tiling: period of the first heritable replicator tape

|                             |   n |   period_median |   tiled |   whole_tape | periods   |   n_tiled |   divides_L |   divides_2L |   gcd_law |
|:----------------------------|----:|----------------:|--------:|-------------:|:----------|----------:|------------:|-------------:|----------:|
| ('lethal@closure', 16, 128) |  10 |               8 |       1 |            0 | 4×4, 8×6  |        10 |           1 |            1 |         0 |

`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.

### Final-soup high-order entropy (bits/byte), median over seeds

|                        |   128 |
|:-----------------------|------:|
| ('lethal@closure', 16) |  5.58 |

## 5. Faithfulness and functional fraction of the final population

|                             |   n |   final_replicator |   final_faithful |   func_rnd_median |   func_rnd_faithful_median |   func_insitu_median |   insitu_informative_runs |   trep_unfaithful |
|:----------------------------|----:|-------------------:|-----------------:|------------------:|---------------------------:|---------------------:|--------------------------:|------------------:|
| ('lethal@closure', 16, 128) |  10 |                 10 |               10 |              0.94 |                       0.94 |                 0.94 |                        10 |                 0 |

## 6. Census family at 300k steps and at the last step

|                             |   n |   stopped_early | family_300k   | family_last   |   stack_takeover | ldir_takeover         |
|:----------------------------|----:|----------------:|:--------------|:--------------|-----------------:|:----------------------|
| ('lethal@closure', 16, 128) |  10 |               0 | ldir:10       | ldir:10       |                0 | 10 runs, median 29500 |
