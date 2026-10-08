# Numbers for the findings (generated; do not edit)

## 1. Emergence by tape length (mutation 1/16)

### mixed@closure

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     16 |                  0 | 10/10             | 10/10           | 10/10 (100)     |

## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)


## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0

|                            |   n |   zero_final_mean |   zero_final_min |   zero_final_max |   zero_emergence_mean |   H0_final_mean |   steps_run_min |
|:---------------------------|----:|------------------:|-----------------:|-----------------:|----------------------:|----------------:|----------------:|
| ('mixed@closure', 16, 128) |  10 |              0.09 |            0.004 |            0.172 |                 0.085 |           3.436 |          300000 |

## 4. Tiling: period of the first heritable replicator tape

|                            |   n |   period_median |   tiled |   whole_tape | periods   |   n_tiled |   divides_L |   divides_2L |   gcd_law |
|:---------------------------|----:|----------------:|--------:|-------------:|:----------|----------:|------------:|-------------:|----------:|
| ('mixed@closure', 16, 128) |  10 |               2 |       1 |            0 | 2×10      |        10 |           1 |            1 |         0 |

`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.

### Final-soup high-order entropy (bits/byte), median over seeds

|                       |   128 |
|:----------------------|------:|
| ('mixed@closure', 16) |  1.86 |

## 5. Faithfulness and functional fraction of the final population

|                            |   n |   final_replicator |   final_faithful |   func_rnd_median |   func_rnd_faithful_median |   func_insitu_median |   insitu_informative_runs |   trep_unfaithful |
|:---------------------------|----:|-------------------:|-----------------:|------------------:|---------------------------:|---------------------:|--------------------------:|------------------:|
| ('mixed@closure', 16, 128) |  10 |                 10 |               10 |              0.67 |                       0.66 |                  0.7 |                        10 |                 0 |

## 6. Census family at 300k steps and at the last step

|                            |   n |   stopped_early | family_300k             | family_last             | stack_takeover      | ldir_takeover      |
|:---------------------------|----:|----------------:|:------------------------|:------------------------|:--------------------|:-------------------|
| ('mixed@closure', 16, 128) |  10 |               0 | push:5, ex_sp:4, ldir:1 | push:5, ex_sp:4, ldir:1 | 10 runs, median 175 | 1 runs, median 350 |
