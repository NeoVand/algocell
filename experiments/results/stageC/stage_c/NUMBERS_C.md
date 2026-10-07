# Stage C numbers (generated)

Interaction clock: mean active interactions per cell per step over all Stage C runs = 0.3949 (min 0.3941, max 0.3959); 1,000 steps ≈ 395 encounters per cell.

## C1 — 1,000,000-step runs

| label    |   steps |   k |   n |   steps_run_min |   tq_10_n |   t_rep_n |   t_rep_km |   t_faith_n | periods   | mechs                            |   func_rnd_median |   zero_final_median |
|:---------|--------:|----:|----:|----------------:|----------:|----------:|-----------:|------------:|:----------|:---------------------------------|------------------:|--------------------:|
| all-ld   |      32 |   2 |  10 |         1000000 |         2 |         2 |        inf |           2 | 4×2       | block-copy:1, block-copy+stack:1 |             0.672 |             0.0527  |
| no-copy  |      32 |   2 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.194   |
| rmw-only |      32 |   2 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.00403 |
| all-ld   |     128 |   4 |  10 |         1000000 |         1 |         2 |        inf |           2 | 8×2       | block-copy+stack:1, -:1          |             0     |             0.327   |
| no-copy  |     128 |   4 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.246   |
| rmw-only |     128 |   4 |  10 |         1000000 |         0 |         0 |        inf |           0 | –         | –                                |             0     |             0.0042  |

## C3 — ablations at 128 steps, mutation 1/16, L = 16, seeds 101–110

| label            |   n |   steps_run_min |   t_rep_n |   t_rep_km |   t_rep_km_ix |   km_ratio_vs_none |   slower |   faster |   ties |    sign_p |   t_faith_n |   tq_10_n |   tiled_frac |   div2L_tiled | periods   |   zero_final_median |
|:-----------------|----:|----------------:|----------:|-----------:|--------------:|-------------------:|---------:|---------:|-------:|----------:|------------:|----------:|-------------:|--------------:|:----------|--------------------:|
| none             |  10 |          300000 |        10 | 200        |     79        |               1    |      nan |      nan |    nan | nan       |          10 |        10 |            1 |             1 | 2×10      |             0.00421 |
| stack-writes     |  10 |          300000 |         8 |   3.2e+04  |      1.26e+04 |             160    |       10 |        0 |      0 |   0.00195 |           8 |         1 |            1 |             1 | 4×4, 8×4  |             0.00615 |
| stack-write-only |  10 |          300000 |         8 |   7.65e+04 |      3.02e+04 |             382    |       10 |        0 |      0 |   0.00195 |           8 |         1 |            1 |             1 | 4×3, 8×5  |             0.00613 |
| stack-read-only  |  10 |          300000 |        10 | 600        |    237        |               3    |        5 |        5 |      0 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.0149  |
| push-only        |  10 |          300000 |         6 |   1.98e+05 |      7.84e+04 |             992    |       10 |        0 |      0 |   0.00195 |           6 |         7 |            1 |             1 | 4×3, 8×3  |             0.0142  |
| ex-sp-only       |  10 |          300000 |        10 | 700        |    277        |               3.5  |        7 |        2 |      1 |   0.18    |          10 |        10 |            1 |             1 | 2×10      |             0.0195  |
| call-rst         |  10 |          300000 |        10 | 250        |     98.6      |               1.25 |        6 |        3 |      1 |   0.508   |          10 |        10 |            1 |             1 | 2×10      |             0.0103  |
| ld-imm           |  10 |          300000 |         2 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           2 |        10 |            1 |             1 | 4×2       |             0.418   |
| ld-reg           |  10 |          300000 |        10 | 150        |     59.2      |               0.75 |        5 |        5 |      0 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.00383 |
| ld-mem           |  10 |          300000 |        10 | 200        |     79        |               1    |        4 |        4 |      2 |   1       |          10 |        10 |            1 |             1 | 2×10      |             0.00388 |
| cb-page          |  10 |          300000 |        10 |   1.3e+03  |    513        |               6.5  |        6 |        4 |      0 |   0.754   |          10 |        10 |            1 |             1 | 2×10      |             0.00402 |
| ed-loads         |  10 |          300000 |        10 | 450        |    178        |               2.25 |        6 |        3 |      1 |   0.508   |          10 |        10 |            1 |             1 | 2×10      |             0.00409 |
| block-copy       |  10 |          300000 |        10 | 350        |    138        |               1.75 |        6 |        4 |      0 |   0.754   |          10 |        10 |            1 |             1 | 2×10      |             0.0038  |
| all-ld           |  10 |         1000000 |         2 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           2 |         1 |            1 |             1 | 8×2       |             0.327   |
| rmw-only         |  10 |         1000000 |         0 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           0 |         0 |          nan |           nan | –         |             0.0042  |
| no-copy          |  10 |         1000000 |         0 | inf        |    inf        |             inf    |       10 |        0 |      0 |   0.00195 |           0 |         0 |          nan |           nan | –         |             0.246   |

`slower/faster/ties`: seed-paired comparison of t_rep against `none` (a censored run counts as slower than any emerged run; two censored runs tie); `sign_p` two-sided exact sign test ignoring ties; `t_rep_km_ix` = KM median in cumulative active interactions per cell.

## C3 — ablations at 32 steps, mutation 1/4, L = 16, seeds 101–110

| label            |   n |   steps_run_min |   t_rep_n |   t_rep_km |   t_rep_km_ix |   km_ratio_vs_none |   slower |   faster |   ties |    sign_p |   t_faith_n |   tq_10_n |   tiled_frac |   div2L_tiled | periods   |   zero_final_median |
|:-----------------|----:|----------------:|----------:|-----------:|--------------:|-------------------:|---------:|---------:|-------:|----------:|------------:|----------:|-------------:|--------------:|:----------|--------------------:|
| none             |  10 |          300000 |         9 |   6.85e+04 |      2.7e+04  |              1     |      nan |      nan |    nan | nan       |           9 |         2 |            1 |             1 | 4×9       |             0.0508  |
| stack-writes     |  10 |          300000 |         5 |   2.28e+05 |      9e+04    |              3.33  |        7 |        3 |      0 |   0.344   |           5 |         0 |            1 |             1 | 4×5       |             0.0106  |
| stack-write-only |  10 |          300000 |        10 |   2.15e+04 |      8.49e+03 |              0.314 |        4 |        6 |      0 |   0.754   |          10 |         1 |            1 |             1 | 4×10      |             0.00929 |
| stack-read-only  |  10 |          300000 |        10 |   9.8e+04  |      3.87e+04 |              1.43  |        6 |        4 |      0 |   0.754   |          10 |         1 |            1 |             1 | 4×10      |             0.0501  |
| push-only        |  10 |          300000 |         9 |   9.3e+04  |      3.67e+04 |              1.36  |        7 |        3 |      0 |   0.344   |           9 |         0 |            1 |             1 | 4×9       |             0.0394  |
| ex-sp-only       |  10 |          300000 |        10 |   7.4e+04  |      2.92e+04 |              1.08  |        4 |        6 |      0 |   0.754   |          10 |         0 |            1 |             1 | 4×10      |             0.0516  |
| call-rst         |  10 |          300000 |         6 |   4.25e+04 |      1.68e+04 |              0.62  |        5 |        4 |      1 |   1       |           6 |         5 |            1 |             1 | 2×4, 4×2  |             0.0245  |
| ld-imm           |  10 |          300000 |         9 |   6.75e+04 |      2.66e+04 |              0.985 |        4 |        5 |      1 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0519  |
| ld-reg           |  10 |          300000 |        10 |   1.04e+05 |      4.13e+04 |              1.53  |        7 |        3 |      0 |   0.344   |          10 |         0 |            1 |             1 | 4×10      |             0.0502  |
| cb-page          |  10 |          300000 |         9 |   9.9e+04  |      3.91e+04 |              1.45  |        5 |        5 |      0 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0515  |
| ed-loads         |  10 |          300000 |         9 |   6.05e+04 |      2.39e+04 |              0.883 |        5 |        5 |      0 |   1       |           9 |         0 |            1 |             1 | 4×9       |             0.0509  |
| all-ld           |  10 |         1000000 |         2 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           2 |         2 |            1 |             1 | 4×2       |             0.0527  |
| rmw-only         |  10 |         1000000 |         0 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           0 |         0 |          nan |           nan | –         |             0.00403 |
| no-copy          |  10 |         1000000 |         0 | inf        |    inf        |            inf     |        9 |        0 |      1 |   0.00391 |           0 |         0 |          nan |           nan | –         |             0.194   |

`slower/faster/ties`: seed-paired comparison of t_rep against `none` (a censored run counts as slower than any emerged run; two censored runs tie); `sign_p` two-sided exact sign test ignoring ties; `t_rep_km_ix` = KM median in cumulative active interactions per cell.

## C2 — succession without censoring (census family of the dominant unit at fixed steps; medians over seeds)

| label      |   steps |   k |   n | family_5k               | family_50k                      | family_300k             |   ldir_300k |   zero8_300k |   final_zero_frac | ldir_takeover         | stack_takeover        |
|:-----------|--------:|----:|----:|:------------------------|:--------------------------------|:------------------------|------------:|-------------:|------------------:|:----------------------|:----------------------|
| block-copy |     128 |   2 |  10 | none:10                 | none:9, ex_sp:1                 | none:6, ex_sp:4         |     7.5e-05 |     0.201    |          0.229    | 0                     | 10 runs, median 14500 |
| block-copy |     128 |   4 |  10 | none:8, push:1, ex_sp:1 | ex_sp:7, none:3                 | ex_sp:10                |     0       |     0.0014   |          0.0038   | 0                     | 9 runs, median 4750   |
| block-copy |     128 |   6 |  10 | none:10                 | push:7, none:2, ex_sp:1         | ex_sp:6, push:4         |     0       |     0.00045  |          0.0013   | 0                     | 9 runs, median 40500  |
| block-copy |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     5e-05   |     0.126    |          0.13     | 0                     | 10 runs, median 1750  |
| block-copy |     512 |   4 |  10 | push:10                 | push:7, ex_sp:3                 | ex_sp:10                |     0       |     0.00753  |          0.0144   | 0                     | 10 runs, median 1150  |
| block-copy |     512 |   6 |  10 | none:9, push:1          | push:8, ex_sp:2                 | push:7, ex_sp:3         |     0       |     0.0651   |          0.0713   | 0                     | 10 runs, median 1825  |
| ld-mem     |     128 |   2 |  10 | none:9, push:1          | push:6, ldir:2, none:2          | ldir:7, ex_sp:3         |     0.794   |     0.0151   |          0.0374   | 7 runs, median 68000  | 9 runs, median 9000   |
| ld-mem     |     128 |   4 |  10 | none:8, push:1, ex_sp:1 | ex_sp:9, none:1                 | ex_sp:10                |     0       |     0.00125  |          0.00388  | 0                     | 10 runs, median 5225  |
| ld-mem     |     128 |   6 |  10 | none:10                 | ex_sp:6, push:4                 | ex_sp:9, ldir:1         |     0       |     0.0003   |          0.000759 | 1 runs, median 53500  | 10 runs, median 21250 |
| ld-mem     |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     2.5e-05 |     0.109    |          0.11     | 0                     | 10 runs, median 1175  |
| ld-mem     |     512 |   4 |  10 | push:7, none:2, ex_sp:1 | ex_sp:9, push:1                 | ex_sp:10                |     0       |     0.00585  |          0.0118   | 0                     | 10 runs, median 1275  |
| ld-mem     |     512 |   6 |  10 | none:9, ldir:1          | ex_sp:5, push:2, ldir:2, none:1 | ex_sp:8, ldir:2         |     0       |     0.00128  |          0.00386  | 2 runs, median 4950   | 9 runs, median 1000   |
| none       |      32 |   2 |  10 | none:10                 | none:7, ldir:3                  | ldir:9, none:1          |     0.84    |     0.0081   |          0.0508   | 9 runs, median 64000  | 0                     |
| none       |     128 |   2 |  10 | none:9, ldir:1          | none:6, ldir:3, ex_sp:1         | ldir:8, ex_sp:2         |     0.803   |     0.0243   |          0.0402   | 8 runs, median 63750  | 9 runs, median 15000  |
| none       |     128 |   4 |  10 | none:10                 | ex_sp:8, ldir:1, none:1         | ex_sp:9, ldir:1         |     0       |     0.00185  |          0.00421  | 1 runs, median 38500  | 9 runs, median 3800   |
| none       |     128 |   6 |  10 | none:10                 | ex_sp:5, push:3, none:1, ldir:1 | ex_sp:8, ldir:2         |     0       |     0.000325 |          0.000914 | 2 runs, median 130750 | 7 runs, median 14500  |
| none       |     512 |   2 |  10 | push:10                 | push:10                         | push:10                 |     0       |     0.13     |          0.134    | 0                     | 10 runs, median 1525  |
| none       |     512 |   4 |  10 | push:9, none:1          | ex_sp:7, push:3                 | ex_sp:9, ldir:1         |     0       |     0.0074   |          0.0142   | 1 runs, median 116500 | 10 runs, median 1625  |
| none       |     512 |   6 |  10 | push:5, none:5          | push:8, ex_sp:2                 | push:6, ex_sp:3, ldir:1 |     0       |     0.0577   |          0.0655   | 1 runs, median 260000 | 10 runs, median 4175  |

### C2 emergence by mutation rate

| label      |   steps |   k |   n |   t_rep_n |   t_rep_km |   t_rep_km_ix |   t_faith_n |   final_faithful_n |   func_rnd_median | periods   |   zero_final_median |
|:-----------|--------:|----:|----:|----------:|-----------:|--------------:|------------:|-------------------:|------------------:|:----------|--------------------:|
| block-copy |     128 |   2 |  10 |        10 | 850        |    336        |          10 |                 10 |             0.219 | 2×10      |            0.229    |
| block-copy |     128 |   4 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.844 | 2×10      |            0.0038   |
| block-copy |     128 |   6 |  10 |        10 |   4.1e+03  |      1.62e+03 |          10 |                 10 |             0.969 | 2×10      |            0.0013   |
| block-copy |     512 |   2 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.281 | 2×10      |            0.13     |
| block-copy |     512 |   4 |  10 |        10 | 250        |     98.8      |          10 |                 10 |             0.766 | 2×10      |            0.0144   |
| block-copy |     512 |   6 |  10 |        10 |   1.05e+03 |    414        |          10 |                 10 |             0.406 | 2×10      |            0.0713   |
| ld-mem     |     128 |   2 |  10 |        10 |   1.15e+03 |    454        |          10 |                 10 |             0.656 | 2×10      |            0.0374   |
| ld-mem     |     128 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.875 | 2×10      |            0.00388  |
| ld-mem     |     128 |   6 |  10 |        10 | 850        |    336        |          10 |                 10 |             0.984 | 2×10      |            0.000759 |
| ld-mem     |     512 |   2 |  10 |        10 | 200        |     79.1      |          10 |                 10 |             0.391 | 2×10      |            0.11     |
| ld-mem     |     512 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.672 | 2×10      |            0.0118   |
| ld-mem     |     512 |   6 |  10 |        10 | 150        |     59.2      |          10 |                 10 |             0.922 | 2×9, 16×1 |            0.00386  |
| none       |      32 |   2 |  10 |         9 |   6.85e+04 |      2.7e+04  |           9 |                  9 |             0.781 | 4×9       |            0.0508   |
| none       |     128 |   2 |  10 |        10 | 650        |    257        |          10 |                 10 |             0.703 | 2×10      |            0.0402   |
| none       |     128 |   4 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.844 | 2×10      |            0.00421  |
| none       |     128 |   6 |  10 |        10 | 200        |     79        |          10 |                 10 |             0.969 | 2×10      |            0.000914 |
| none       |     512 |   2 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.344 | 2×10      |            0.134    |
| none       |     512 |   4 |  10 |        10 | 350        |    138        |          10 |                 10 |             0.766 | 2×10      |            0.0142   |
| none       |     512 |   6 |  10 |        10 | 450        |    178        |          10 |                 10 |             0.422 | 2×10      |            0.0655   |

## C5 — L ∈ {36, 100} without early stop (periods of the first heritable replicator and of the faithful final dominant)

| label        |   L |   steps |   k |   n |   t_rep_n |   t_rep_km |   t_faith_n |   final_faithful_n | periods                    | trep_period_le8   | final_period_le8   | final_periods   |
|:-------------|----:|--------:|----:|----:|----------:|-----------:|------------:|-------------------:|:---------------------------|:------------------|:-------------------|:----------------|
| none         |  36 |     128 |   4 |  10 |        10 |  350       |          10 |                 10 | 2×10                       | 10/10             | 3/10               | 2×3, 14×7       |
| none         | 100 |     128 |   4 |  10 |        10 |    1.2e+03 |           9 |                  8 | 2×9, 20×1                  | 9/10              | 0/8                | 9×1, 10×7       |
| stack-writes |  36 |     128 |   4 |  10 |        10 |    1.2e+04 |          10 |                 10 | 4×1, 6×2, 8×3, 9×2, 12×2   | 6/10              | 10/10              | 4×6, 6×2, 8×2   |
| stack-writes | 100 |     128 |   4 |  10 |        10 |    8.5e+03 |           1 |                  1 | 5×1, 8×1, 20×4, 25×3, 76×1 | 2/10              | 0/1                | 9×1             |

C1 pooled: no-copy + rmw-only 0/40 runs emerged within 1,000,000 steps → 95% upper bound on the per-run emergence probability within that horizon = 0.072 (exact binomial, one-sided).

## First replicator vs final dominant under `none` (128 steps, 1/16): the modal tapes assayed in isolation

|   L | which                     | identical_in_seeds   |   period | tape                                                                                                        |   score |   gen2 | faithful   |   partner_bytes_changed |   self_bytes_changed |
|----:|:--------------------------|:---------------------|---------:|:------------------------------------------------------------------------------------------------------------|--------:|-------:|:-----------|------------------------:|---------------------:|
|  16 | first (t_rep)             | 8/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5                                                             |    0.82 |   0.59 | True       |                   15.62 |                 4.50 |
|  16 | final (faithful dominant) | 9/10                 |        8 | ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0                                                             |    1.00 |   1.00 | True       |                   15.94 |                 0.00 |
|  36 | first (t_rep)             | 7/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 |    0.83 |   0.64 | True       |                   34.12 |                 8.08 |
|  36 | final (faithful dominant) | 7/10                 |       14 | 11 d5 11 d5 11 d5 11 d5 11 10 f0 d5 11 d5 11 d5 11 d5 11 d5 11 d5 11 10 f0 d5 11 d5 11 d5 11 d5 11 d5 11 d5 |    0.94 |   0.83 | True       |                   35.81 |                 0.00 |
| 100 | first (t_rep)             | 9/10                 |        2 | 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 …                                               |    0.86 |   0.72 | True       |                   91.95 |                 1.83 |
| 100 | final (faithful dominant) | 7/10                 |       10 | c2 c5 01 c5 01 c5 01 c5 c2 c5 c2 c5 01 c5 01 c5 01 c5 c2 c5 …                                               |    0.99 |   0.99 | True       |                   98.67 |                 2.00 |

`identical_in_seeds`: seeds whose tape is byte-identical to the modal one; `partner_bytes_changed` / `self_bytes_changed`: mean bytes of the random partner / of the tape itself that differ after one 128-step encounter as program A (64 partners).

## Pre-registered C predictions — the numbers

- C1 no-copy @(128, k4): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 no-copy @(32, k2): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 rmw-only @(128, k4): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 rmw-only @(32, k2): heritable 0/10 at 1,000,000 steps (KM median NR); faithful 0/10
- C1 all-ld @(128, k4): heritable 2/10 at 1,000,000 steps (KM median NR); faithful 2/10
- C1 all-ld @(32, k2): heritable 2/10 at 1,000,000 steps (KM median NR); faithful 2/10
- C3 stack-writes @(128, k4): 8/10 heritable, KM 32,000 vs none 200 (ratio 160.00); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 stack-write-only @(128, k4): 8/10 heritable, KM 76,500 vs none 200 (ratio 382.50); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 stack-read-only @(128, k4): 10/10 heritable, KM 600 vs none 200 (ratio 3.00); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 push-only @(128, k4): 6/10 heritable, KM 198,500 vs none 200 (ratio 992.50); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 ex-sp-only @(128, k4): 10/10 heritable, KM 700 vs none 200 (ratio 3.50); paired vs none: 7 slower, 2 faster, 1 ties, sign p = 0.18
- C3 call-rst @(128, k4): 10/10 heritable, KM 250 vs none 200 (ratio 1.25); paired vs none: 6 slower, 3 faster, 1 ties, sign p = 0.508
- C3 ld-imm @(128, k4): 2/10 heritable, KM NR vs none 200 (ratio nan); paired vs none: 10 slower, 0 faster, 0 ties, sign p = 0.00195
- C3 ld-reg @(128, k4): 10/10 heritable, KM 150 vs none 200 (ratio 0.75); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 ld-mem @(128, k4): 10/10 heritable, KM 200 vs none 200 (ratio 1.00); paired vs none: 4 slower, 4 faster, 2 ties, sign p = 1
- C3 cb-page @(128, k4): 10/10 heritable, KM 1,300 vs none 200 (ratio 6.50); paired vs none: 6 slower, 4 faster, 0 ties, sign p = 0.754
- C3 ed-loads @(128, k4): 10/10 heritable, KM 450 vs none 200 (ratio 2.25); paired vs none: 6 slower, 3 faster, 1 ties, sign p = 0.508
- C3 stack-writes @(32, k2): 5/10 heritable, KM 228,000 vs none 68,500 (ratio 3.33); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 stack-write-only @(32, k2): 10/10 heritable, KM 21,500 vs none 68,500 (ratio 0.31); paired vs none: 4 slower, 6 faster, 0 ties, sign p = 0.754
- C3 stack-read-only @(32, k2): 10/10 heritable, KM 98,000 vs none 68,500 (ratio 1.43); paired vs none: 6 slower, 4 faster, 0 ties, sign p = 0.754
- C3 push-only @(32, k2): 9/10 heritable, KM 93,000 vs none 68,500 (ratio 1.36); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 ex-sp-only @(32, k2): 10/10 heritable, KM 74,000 vs none 68,500 (ratio 1.08); paired vs none: 4 slower, 6 faster, 0 ties, sign p = 0.754
- C3 call-rst @(32, k2): 6/10 heritable, KM 42,500 vs none 68,500 (ratio 0.62); paired vs none: 5 slower, 4 faster, 1 ties, sign p = 1
- C3 ld-imm @(32, k2): 9/10 heritable, KM 67,500 vs none 68,500 (ratio 0.99); paired vs none: 4 slower, 5 faster, 1 ties, sign p = 1
- C3 ld-reg @(32, k2): 10/10 heritable, KM 104,500 vs none 68,500 (ratio 1.53); paired vs none: 7 slower, 3 faster, 0 ties, sign p = 0.344
- C3 cb-page @(32, k2): 9/10 heritable, KM 99,000 vs none 68,500 (ratio 1.45); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
- C3 ed-loads @(32, k2): 9/10 heritable, KM 60,500 vs none 68,500 (ratio 0.88); paired vs none: 5 slower, 5 faster, 0 ties, sign p = 1
