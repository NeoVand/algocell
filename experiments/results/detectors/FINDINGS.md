# Life detectors versus assay ground truth, resolved in time and in tape length (2026-10-07)

What was done (`detectors.py`, no new runs): every statistic the runner records per sample — high-order entropy `hoe` (byte entropy minus brotli bits per byte), the non-unique tape fraction, the top tape's exact share, quasispecies occupancy (`q_share`, and its shift-tolerant form), motif share, the fraction of interactions writing ≥ L bytes into the partner (`ix_wb_ge_L_frac`), species entropy, brotli bits per byte, the silent-interaction fraction — was scored as a detector of life against two ground truths at the snapshot steps (500 … 300k; 1M for the nulls): (i) **heritable**: the C4 fraction of 16 random cells with gen2 ≥ 0.3 is ≥ 0.25 (Stage C, 7,960 run×step samples); (ii) **event**: the run's first heritable top-3 exemplar has appeared (Stages C and E). Orientation of each detector was fixed before any AUC was computed (lower species entropy, lower bits per byte and fewer silent interactions count as "alive"). AUC is the Mann–Whitney statistic; samples of one run are correlated, so the per-L and per-step strata are the ones to read. Numbers: `NUMBERS_DETECTORS.md`; samples: `samples_C.csv`, `samples_E.csv`.

## Against the heritable fraction (Stage C)

| detector | pooled | L = 16 | L = 36 | L = 100 |
|---|---|---|---|---|
| hoe | 0.973 | 0.984 | 0.951 | 0.795 |
| non-unique fraction | 0.967 | 0.968 | 0.994 | 0.957 |
| brotli bits/byte (lower) | 0.970 | 0.967 | 0.991 | 0.957 |
| silent-interaction fraction (lower) | 0.949 | 0.943 | 0.994 | 0.971 |
| interactions writing ≥ L bytes | 0.937 | 0.987 | 0.994 | 0.545 |
| species entropy (lower) | 0.900 | 0.894 | 0.989 | 0.940 |
| top tape share | 0.749 | 0.745 | 0.946 | 0.767 |
| quasispecies occupancy | 0.708 | 0.699 | 0.950 | 0.613 |
| motif share | 0.603 | 0.606 | 0.802 | 0.441 |

By step at L = 16 · 128 steps · 1/16 (all C3 arms): the compression-type detectors (hoe, non-unique, bits/byte, species entropy) and the two interaction-based detectors are at 0.82–0.95 from 1,000 to 20,000 steps and at 0.99–1.00 from 30,000 on; the occupancy detectors (top share, quasispecies occupancy, motif share) stay at 0.55–0.72 at every step and fall below 0.5 at 500k–1M. Occupancy does not track heritability because the pusher occupies the soup for tens of thousands of steps while only 7–14% of cells are heritable (C4). The mechanistic detector `ix_wb_ge_L_frac` is the best at L = 16 (0.987) and fails at L = 100 (0.545): at 128 steps the pusher writes 64 bytes per encounter, less than one tape, so "writes ≥ L bytes" never fires there.

## Against the event, by tape length (Stage E @nominal)

Pooled over both ablations, the non-unique fraction, bits/byte and species entropy are ≥ 0.84 at every L from 8 to 100 except L = 81 (0.63–0.73), top share is ≥ 0.95 at every L ≥ 16, and hoe falls to 0.518 (L = 24), 0.674 (49), 0.474 (81), 0.606 (100). Split by ablation the picture is sharper: under `none`, where the pusher appears within ≈ 1,000 steps and the only non-alive samples are the flooded soups at 500–1,000 steps, **hoe is anti-correlated with life** at L = 25 (0.258), 49 (0.065), 64 (0.201), 81 (0.340) and 100 (0.538): the zero flood compresses better than the replicating soup. Under `stack-write-only` (slow LDIR emergence, many non-alive samples) hoe is 0.95–1.00 up to L = 20 and 0.52–0.85 from L = 24 up, while the non-unique fraction stays 0.85–1.00 and top share 0.92–1.00 at every L ≥ 8. At L = 9–12 the occupancy detectors fail in the other direction (`q_share` 0.11–0.19 at L = 9–10): the quasispecies radius ⌈L/4⌉ = 3 is too tight for the LDIR units there.

## What this adds to the paper

The compression-based biosignature (the Computational Life paper's high-order entropy) is an excellent detector of a heritable population at L = 16 and a poor or inverted one wherever a sterile byte flood precedes life or the organism is long; the non-unique tape fraction and brotli bits per byte are the robust compression-type detectors across L; occupancy of a dominant genotype is a poor proxy for heritability at every step; and the interaction-level detector works exactly when a single encounter can write a whole tape. Every detector statement here is relative to the assay ground truth, which is itself a population measure (16 cells × 32 partners per sample).
