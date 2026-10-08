# Invasion assay — can a known replicator take over a soup it did not evolve in? (80 local soups, 2026-10-08)

Design (`invasion.py`, no Modal cost): L = 16, 20,000 cells, `none`, 128 or 32 Z80 steps × mutation 1/4 or 1/16; the pusher `01 c5 ×8` or the LDIR unit `04 5e ed b0 ×4` written into 1% of cells (200) either at step 0 (random soup) or at step 1,000 (after the zero flood has formed); 5 soups per cell, 20,000 steps. Recorded every 250 steps: cells within Hamming distance 4 of the unit (`near4`), cells matching it at ≥ 75% of bytes under some shift, zero fraction; every 2,500 steps the heritable fraction of 16 random cells (assay_many, 32 partners). Numbers: `NUMBERS_INVASION.md`, `invasion_summary.csv`, `invasion.csv`. Motivation: at 32 steps · 1/4 no soup in Stages A, C or E ever produced the pusher (E7: 18/20 LDIR vs 0/20 without block copy) although the pusher is heritable in isolation at that budget (gen2 0.54).

| unit | steps | mutation | seeded at | peak `near4` (median) | at 20k: `near4` | zero fraction | heritable fraction | soups alive (≥ 0.25) |
|---|---|---|---|---|---|---|---|---|
| pusher | 32 | 1/4 | 0 | 0.010 (the seed itself) | 0.000 | 0.328 | 0.000 | 0/5 |
| pusher | 32 | 1/4 | 1,000 | 0.010 | 0.000 | 0.327 | 0.000 | 0/5 |
| pusher | 32 | 1/16 | 0 | 0.010 | 0.000 | 0.292 | 0.062 | 2/5 |
| pusher | 32 | 1/16 | 1,000 | 0.010 | 0.000 | 0.304 | 0.062 | 1/5 |
| pusher | 128 | 1/16 | 0 | 0.319 at 250 | 0.000 | 0.032 | 0.875 | 4/5 |
| pusher | 128 | 1/16 | 1,000 | 0.212 at 1,250 | 0.075 | 0.148 | 0.125 | 1/5 |
| pusher | 128 | 1/4 | 0 | 0.179 at 2,250 | 0.057 | 0.225 | 0.312 | 3/5 |
| pusher | 128 | 1/4 | 1,000 | 0.158 at 2,750 | 0.044 | 0.230 | 0.000 | 1/5 |
| LDIR-4 | 32 | 1/4 | 0 | 0.139 at 250 | 0.015 | 0.050 | 0.750 | 5/5 |
| LDIR-4 | 32 | 1/4 | 1,000 | 0.132 at 1,250 | 0.014 | 0.048 | 0.750 | 5/5 |
| LDIR-4 | 32 | 1/16 | 0 | 0.404 at 250 | 0.013 | 0.051 | 0.938 | 5/5 |
| LDIR-4 | 32 | 1/16 | 1,000 | 0.466 at 1,250 | 0.011 | 0.047 | 1.000 | 5/5 |
| LDIR-4 | 128 | 1/4 | 0 | 0.182 at 250 | 0.004 | 0.038 | 0.688 | 5/5 |
| LDIR-4 | 128 | 1/4 | 1,000 | 0.188 at 1,250 | 0.006 | 0.039 | 0.688 | 5/5 |
| LDIR-4 | 128 | 1/16 | 0 | 0.372 at 250 | 0.026 | 0.020 | 0.938 | 5/5 |
| LDIR-4 | 128 | 1/16 | 1,000 | 0.378 at 1,250 | 0.022 | 0.018 | 0.938 | 5/5 |

## Reading

- **At 32 steps the pusher is unviable, not undiscovered.** Handed to 1% of the cells it never grows (peak occupancy = the seeded 1%), is gone by 20k steps, and the soup floods (zero fraction 0.29–0.33) in 20/20 soups at both mutation rates. The LDIR unit handed to the same soups invades at once (peak 0.14–0.47 at 250–1,250 steps), then diversifies into other LDIR variants (`near4` falls to 0.01 while 75–100% of random cells stay heritable) and keeps the flood at 0.05. This resolves the 32-step puzzle of Stages C and E: the single-encounter assay against random partners (gen2 0.54) overstates the pusher at 32 steps; in the population it cannot maintain itself.
- **At 128 steps · 1/16 the seeded pusher does what the natural one does**: a wave to 0.32 occupancy at 250 steps, then replacement by a heritable population (0.875 of random cells at 20k) in which the pusher pattern is gone — the same succession as in Stage C, compressed in time. Seeding after the flood (step 1,000) gave a smaller wave (0.21) and a less heritable soup at 20k (0.125, flood 0.15), so the flood does not help the pusher; this cell is confounded by the soup's own emergence (natural pushers appear at ≈ 200 steps under these settings) and should not be over-read.
- **At 128 steps · 1/4 the pusher persists at low occupancy** (peak 0.16–0.18 at 2,250–2,750 steps, 0.04–0.06 at 20k) with a persistent flood (0.23) and a population that is only partly heritable (0.31 median, 3/5 soups), consistent with the LDIR take-over seen in Stage C at this mutation rate.
- **The LDIR unit is an invader everywhere**: 20/20 soups alive at 20k, flood suppressed to 0.02–0.05 in every cell.

## Limits

Five soups per cell; `near4` tracks the seeded tape and its immediate mutants only, so a unit that evolves (as LDIR-4 does within 2,000 steps) disappears from it while the population stays alive — read `near4` with the heritable fraction. The 128-step cells are confounded with natural emergence and are reported for completeness. Horizon 20,000 steps.
