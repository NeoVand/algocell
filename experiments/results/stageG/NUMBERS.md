# Numbers for the findings (generated; do not edit)

## 1. Emergence by tape length (mutation 1/16)

### none@closure

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     16 |                  0 | 20/20             | 20/20           | 20/20 (450)     |
|     50 |                  0 | 20/20             | 20/20           | 20/20 (600)     |

### none@closure1M

|   tape |   ('stopped', 128) | ('tfaith', 128)   | ('tq10', 128)   | ('trep', 128)   |
|-------:|-------------------:|:------------------|:----------------|:----------------|
|     20 |                  0 | 20/20             | 19/20           | 20/20 (700)     |
|     64 |                  0 | 20/20             | 20/20           | 20/20 (750)     |

## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)


## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0

|                             |   n |   zero_final_mean |   zero_final_min |   zero_final_max |   zero_emergence_mean |   H0_final_mean |   steps_run_min |
|:----------------------------|----:|------------------:|-----------------:|-----------------:|----------------------:|----------------:|----------------:|
| ('none@closure', 16, 128)   |  20 |             0.006 |            0.003 |            0.019 |                 0.268 |           3.254 |      300000     |
| ('none@closure', 50, 128)   |  20 |             0.049 |            0.045 |            0.055 |                 0.337 |           2.206 |      300000     |
| ('none@closure1M', 20, 128) |  20 |             0.034 |            0.001 |            0.223 |                 0.166 |           3.923 |           1e+06 |
| ('none@closure1M', 64, 128) |  20 |             0.052 |            0.007 |            0.091 |                 0.282 |           3.815 |           1e+06 |

## 4. Tiling: period of the first heritable replicator tape

|                             |   n |   period_median |   tiled |   whole_tape | periods   |   n_tiled |   divides_L |   divides_2L |   gcd_law |
|:----------------------------|----:|----------------:|--------:|-------------:|:----------|----------:|------------:|-------------:|----------:|
| ('none@closure', 16, 128)   |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure', 50, 128)   |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure1M', 20, 128) |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |
| ('none@closure1M', 64, 128) |  20 |               2 |       1 |            0 | 2×20      |        20 |           1 |            1 |         0 |

`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.

### Final-soup high-order entropy (bits/byte), median over seeds

|                        |   128 |
|:-----------------------|------:|
| ('none@closure', 16)   |  2.09 |
| ('none@closure', 50)   |  1.26 |
| ('none@closure1M', 20) |  2.72 |
| ('none@closure1M', 64) |  1.97 |

## 5. Faithfulness and functional fraction of the final population

|                             |   n |   final_replicator |   final_faithful |   func_rnd_median |   func_rnd_faithful_median |   func_insitu_median |   insitu_informative_runs |   trep_unfaithful |
|:----------------------------|----:|-------------------:|-----------------:|------------------:|---------------------------:|---------------------:|--------------------------:|------------------:|
| ('none@closure', 16, 128)   |  20 |                 20 |               20 |              0.86 |                       0.86 |                 0.35 |                        20 |                 0 |
| ('none@closure', 50, 128)   |  20 |                 20 |               20 |              0.56 |                       0.41 |                 0.36 |                        20 |                 0 |
| ('none@closure1M', 20, 128) |  20 |                 20 |               20 |              0.95 |                       0.95 |                 0.9  |                        20 |                 0 |
| ('none@closure1M', 64, 128) |  20 |                 20 |               20 |              0.34 |                       0.22 |                 0.27 |                        20 |                 0 |

## 6. Census family at 300k steps and at the last step

|                             |   n |   stopped_early | family_300k             | family_last             | stack_takeover        | ldir_takeover          |
|:----------------------------|----:|----------------:|:------------------------|:------------------------|:----------------------|:-----------------------|
| ('none@closure', 16, 128)   |  20 |               0 | ex_sp:17, ldir:3        | ex_sp:17, ldir:3        | 17 runs, median 8500  | 3 runs, median 43500   |
| ('none@closure', 50, 128)   |  20 |               0 | push:20                 | push:20                 | 20 runs, median 1350  | 0                      |
| ('none@closure1M', 20, 128) |  20 |               0 | ldir:10, push:5, none:5 | ldir:16, push:3, none:1 | 17 runs, median 18000 | 16 runs, median 247500 |
| ('none@closure1M', 64, 128) |  20 |               0 | push:17, ldir:3         | push:13, ldir:7         | 20 runs, median 1750  | 7 runs, median 419000  |
