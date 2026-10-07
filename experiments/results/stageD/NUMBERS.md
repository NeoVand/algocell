# Numbers for the findings (generated; do not edit)

## 1. Emergence by tape length (mutation 1/16)

### call-rst-write

|   tape |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 128)   | ('trep', 512)   |
|-------:|-------------------:|-------------------:|:------------------|:------------------|:----------------|:----------------|:----------------|:----------------|
|      9 |                  0 |                  0 | 12/20             | 14/20             | 18/20           | 12/20           | 12/20 (192,000) | 14/20 (160,500) |

### none

|   tape |   ('stopped', 32) |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 32)   | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 32)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 32)   | ('trep', 128)   | ('trep', 512)   |
|-------:|------------------:|-------------------:|-------------------:|:-----------------|:------------------|:------------------|:---------------|:----------------|:----------------|:---------------|:----------------|:----------------|
|      9 |                 0 |                  0 |                  0 | 4/20             | 6/20              | 5/20              | 20/20          | 20/20           | 15/20           | 4/20 (NR)      | 6/20 (NR)       | 5/20 (NR)       |

### push

|   tape |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 128)   | ('trep', 512)   |
|-------:|-------------------:|-------------------:|:------------------|:------------------|:----------------|:----------------|:----------------|:----------------|
|      9 |                  0 |                  0 | 11/20             | 8/20              | 10/20           | 8/20            | 11/20 (226,500) | 8/20 (NR)       |

### stack-read-only

|   tape |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 128)   | ('trep', 512)   |
|-------:|-------------------:|-------------------:|:------------------|:------------------|:----------------|:----------------|:----------------|:----------------|
|      9 |                  0 |                  0 | 8/20              | 10/20             | 20/20           | 14/20           | 8/20 (NR)       | 10/20 (252,000) |

### stack-write-only

|   tape |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 128)   | ('trep', 512)   |
|-------:|-------------------:|-------------------:|:------------------|:------------------|:----------------|:----------------|:----------------|:----------------|
|      9 |                  0 |                  0 | 12/20             | 19/20             | 9/20            | 16/20           | 12/20 (138,500) | 19/20 (90,000)  |

### stack-writes

|   tape |   ('stopped', 128) |   ('stopped', 512) | ('tfaith', 128)   | ('tfaith', 512)   | ('tq10', 128)   | ('tq10', 512)   | ('trep', 128)   | ('trep', 512)   |
|-------:|-------------------:|-------------------:|:------------------|:------------------|:----------------|:----------------|:----------------|:----------------|
|      9 |                  0 |                  0 | 12/20             | 18/20             | 11/20           | 16/20           | 12/20 (152,000) | 18/20 (82,500)  |

## 2. L = 9: none vs stack-writes (post hoc; one cell of the design)

- 128 steps, t_rep: stack-writes 12/20 vs none 6/20; Fisher two-sided p = 0.111, one-sided (stack-writes greater) p = 0.0555; KM medians 152,000 vs NR
- 128 steps, t_faith: stack-writes 12/20 vs none 6/20; Fisher two-sided p = 0.111, one-sided (stack-writes greater) p = 0.0555; KM medians 152,000 vs NR
- 128 steps, tq_10: stack-writes 11/20 vs none 20/20; Fisher two-sided p = 0.00123, one-sided (stack-writes greater) p = 1; KM medians 210,500 vs 100
- 512 steps, t_rep: stack-writes 18/20 vs none 5/20; Fisher two-sided p = 6.86e-05, one-sided (stack-writes greater) p = 3.43e-05; KM medians 82,500 vs NR
- 512 steps, t_faith: stack-writes 18/20 vs none 5/20; Fisher two-sided p = 6.86e-05, one-sided (stack-writes greater) p = 3.43e-05; KM medians 82,500 vs NR
- 512 steps, tq_10: stack-writes 16/20 vs none 15/20; Fisher two-sided p = 1, one-sided (stack-writes greater) p = 0.5; KM medians 131,000 vs 100

## 3. Zero-byte fraction of the soup (from snapshots) and final byte entropy H0

|                              |   n |   zero_final_mean |   zero_final_min |   zero_final_max |   zero_emergence_mean |   H0_final_mean |   steps_run_min |
|:-----------------------------|----:|------------------:|-----------------:|-----------------:|----------------------:|----------------:|----------------:|
| ('call-rst-write', 9, 128)   |  20 |             0.12  |            0.008 |            0.285 |                 0.252 |           6.408 |          300000 |
| ('call-rst-write', 9, 512)   |  20 |             0.091 |            0.007 |            0.274 |                 0.199 |           6.342 |          300000 |
| ('none', 9, 32)              |  20 |             0.308 |            0.013 |            0.383 |                 0.372 |           5.783 |          300000 |
| ('none', 9, 128)             |  40 |             0.276 |            0.005 |            0.362 |                 0.317 |           5.687 |          300000 |
| ('none', 9, 512)             |  20 |             0.251 |            0.014 |            0.33  |                 0.313 |           6.119 |          300000 |
| ('push', 9, 128)             |  20 |             0.133 |            0.009 |            0.302 |                 0.181 |           6.062 |          300000 |
| ('push', 9, 512)             |  20 |             0.165 |            0.011 |            0.289 |                 0.237 |           6.082 |          300000 |
| ('stack-read-only', 9, 128)  |  20 |             0.21  |            0.013 |            0.341 |                 0.32  |           6.064 |          300000 |
| ('stack-read-only', 9, 512)  |  20 |             0.171 |            0.012 |            0.327 |                 0.283 |           6.207 |          300000 |
| ('stack-write-only', 9, 128) |  20 |             0.094 |            0.004 |            0.196 |                 0.166 |           6.481 |          300000 |
| ('stack-write-only', 9, 512) |  20 |             0.021 |            0.004 |            0.187 |                 0.162 |           6.198 |          300000 |
| ('stack-writes', 9, 128)     |  20 |             0.072 |            0.005 |            0.186 |                 0.141 |           6.475 |          300000 |
| ('stack-writes', 9, 512)     |  20 |             0.03  |            0.003 |            0.18  |                 0.145 |           6.111 |          300000 |

## 4. Tiling: period of the first heritable replicator tape

|                              |   n |   period_median |   tiled |   whole_tape | periods       |   n_tiled |   divides_L |   divides_2L |   gcd_law |
|:-----------------------------|----:|----------------:|--------:|-------------:|:--------------|----------:|------------:|-------------:|----------:|
| ('call-rst-write', 9, 128)   |  12 |               3 |    0.67 |            3 | 3×8, 6×1, 9×3 |         8 |           1 |            1 |         0 |
| ('call-rst-write', 9, 512)   |  14 |               3 |    0.71 |            4 | 3×10, 9×4     |        10 |           1 |            1 |         0 |
| ('none', 9, 32)              |   4 |               3 |    1    |            0 | 3×4           |         4 |           1 |            1 |         0 |
| ('none', 9, 128)             |   9 |               3 |    0.89 |            1 | 3×8, 9×1      |         8 |           1 |            1 |         0 |
| ('none', 9, 512)             |   5 |               3 |    0.8  |            1 | 3×4, 9×1      |         4 |           1 |            1 |         0 |
| ('push', 9, 128)             |  11 |               3 |    0.82 |            2 | 3×9, 9×2      |         9 |           1 |            1 |         0 |
| ('push', 9, 512)             |   8 |               3 |    1    |            0 | 3×8           |         8 |           1 |            1 |         0 |
| ('stack-read-only', 9, 128)  |   8 |               3 |    1    |            0 | 3×8           |         8 |           1 |            1 |         0 |
| ('stack-read-only', 9, 512)  |  10 |               3 |    0.8  |            2 | 3×8, 9×2      |         8 |           1 |            1 |         0 |
| ('stack-write-only', 9, 128) |  12 |               3 |    0.92 |            1 | 3×11, 9×1     |        11 |           1 |            1 |         0 |
| ('stack-write-only', 9, 512) |  19 |               3 |    0.84 |            3 | 3×16, 9×3     |        16 |           1 |            1 |         0 |
| ('stack-writes', 9, 128)     |  12 |               3 |    0.75 |            3 | 3×9, 9×3      |         9 |           1 |            1 |         0 |
| ('stack-writes', 9, 512)     |  18 |               3 |    0.89 |            2 | 3×16, 9×2     |        16 |           1 |            1 |         0 |

`tiled` = period ≤ L/2; `whole_tape` = period L with copy offset 0 (exact whole-tape copier); `divides_*` and `gcd_law` (period == gcd(offset, 2L)) are computed over tiled tapes only.

### Final-soup high-order entropy (bits/byte), median over seeds

|                         |     32 |   128 |   512 |
|:------------------------|-------:|------:|------:|
| ('call-rst-write', 9)   | nan    |  4.61 |  4.43 |
| ('none', 9)             |  -0.15 | -0.09 | -0.05 |
| ('push', 9)             | nan    |  4.14 | -0.01 |
| ('stack-read-only', 9)  | nan    | -0.07 |  2.09 |
| ('stack-write-only', 9) | nan    |  3.54 |  4.6  |
| ('stack-writes', 9)     | nan    |  4.27 |  4.59 |

## 5. Faithfulness and functional fraction of the final population

|                              |   n |   final_replicator |   final_faithful |   func_rnd_median |   func_rnd_faithful_median |   func_insitu_median |   insitu_informative_runs |   trep_unfaithful |
|:-----------------------------|----:|-------------------:|-----------------:|------------------:|---------------------------:|---------------------:|--------------------------:|------------------:|
| ('call-rst-write', 9, 128)   |  20 |                 12 |               12 |              0.83 |                       0.83 |                 0.86 |                        20 |                 1 |
| ('call-rst-write', 9, 512)   |  20 |                 14 |               14 |              0.83 |                       0.83 |                 0.83 |                        20 |                 0 |
| ('none', 9, 32)              |  20 |                  4 |                4 |              0    |                       0    |                 0    |                        20 |                 0 |
| ('none', 9, 128)             |  40 |                  9 |                9 |              0    |                       0    |                 0    |                        40 |                 0 |
| ('none', 9, 512)             |  20 |                  5 |                5 |              0    |                       0    |                 0    |                        20 |                 0 |
| ('push', 9, 128)             |  20 |                 12 |               12 |              0.78 |                       0.78 |                 0.81 |                        20 |                 0 |
| ('push', 9, 512)             |  20 |                  9 |                9 |              0    |                       0    |                 0    |                        20 |                 0 |
| ('stack-read-only', 9, 128)  |  20 |                  8 |                8 |              0    |                       0    |                 0    |                        20 |                 0 |
| ('stack-read-only', 9, 512)  |  20 |                 10 |               10 |              0.42 |                       0.42 |                 0.42 |                        20 |                 0 |
| ('stack-write-only', 9, 128) |  20 |                 12 |               12 |              0.8  |                       0.8  |                 0.83 |                        20 |                 0 |
| ('stack-write-only', 9, 512) |  20 |                 19 |               19 |              0.88 |                       0.88 |                 0.91 |                        20 |                 0 |
| ('stack-writes', 9, 128)     |  20 |                 13 |               13 |              0.84 |                       0.84 |                 0.86 |                        20 |                 0 |
| ('stack-writes', 9, 512)     |  20 |                 18 |               18 |              0.86 |                       0.86 |                 0.89 |                        20 |                 0 |

## 6. Census family at 300k steps and at the last step

|                              |   n |   stopped_early | family_300k      | family_last      |   stack_takeover | ldir_takeover          |
|:-----------------------------|----:|----------------:|:-----------------|:-----------------|-----------------:|:-----------------------|
| ('call-rst-write', 9, 128)   |  20 |               0 | ldir:12, none:8  | ldir:12, none:8  |                0 | 12 runs, median 106000 |
| ('call-rst-write', 9, 512)   |  20 |               0 | ldir:14, none:6  | ldir:14, none:6  |                0 | 14 runs, median 95250  |
| ('none', 9, 32)              |  20 |               0 | none:16, ldir:4  | none:16, ldir:4  |                0 | 4 runs, median 55750   |
| ('none', 9, 128)             |  40 |               0 | none:31, ldir:9  | none:31, ldir:9  |                0 | 9 runs, median 121000  |
| ('none', 9, 512)             |  20 |               0 | none:15, ldir:5  | none:15, ldir:5  |                0 | 5 runs, median 96000   |
| ('push', 9, 128)             |  20 |               0 | ldir:12, none:8  | ldir:12, none:8  |                0 | 12 runs, median 144500 |
| ('push', 9, 512)             |  20 |               0 | none:11, ldir:9  | none:11, ldir:9  |                0 | 9 runs, median 135500  |
| ('stack-read-only', 9, 128)  |  20 |               0 | none:12, ldir:8  | none:12, ldir:8  |                0 | 8 runs, median 103000  |
| ('stack-read-only', 9, 512)  |  20 |               0 | ldir:10, none:10 | ldir:10, none:10 |                0 | 10 runs, median 63250  |
| ('stack-write-only', 9, 128) |  20 |               0 | ldir:11, none:9  | ldir:11, none:9  |                0 | 11 runs, median 106500 |
| ('stack-write-only', 9, 512) |  20 |               0 | ldir:19, none:1  | ldir:19, none:1  |                0 | 19 runs, median 91500  |
| ('stack-writes', 9, 128)     |  20 |               0 | ldir:13, none:7  | ldir:13, none:7  |                0 | 13 runs, median 85500  |
| ('stack-writes', 9, 512)     |  20 |               0 | ldir:18, none:2  | ldir:18, none:2  |                0 | 18 runs, median 82750  |
