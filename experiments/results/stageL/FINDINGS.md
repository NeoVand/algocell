# Stage L — ten-million-step extensions (2026-10-09; pre-registered in `REVISION_PREREG.md`, L)

Stage G conditions at L = 16 and 20, ten worlds each (seeds 6001–6010), ten million steps, Modal batch `stageL`
(≈ 1,300 s per world on an L40S; ≈ $8). Scored by the Stage G pipeline (`stage_pipeline.sh`; tables in `stageL/`) and by
the capacity scan of the most common tape at every snapshot (`capacity_over_time.py --stages L`;
`results/capacity_time_L/capacity_over_time.csv`); scoring `score_L.py` → `SCORING.md`, `scoring_L.csv`.

| prediction | outcome |
|---|---|
| L1 recovery: more transmissible sites at 10M than the closed dominant at 300k (L = 16) / 1M (L = 20) in ≥ 5/10 at each length | **not met**: 1/10 (L = 16), 0/10 (L = 20) |
| L2 every recovery in a block-copy lineage through copied-but-unexecuted sites | **not met** (L = 16: the one recovery, 0 → 4 sites, is a block-copy lineage with 1 of 4 sites unexecuted); n/a at L = 20 |
| L3 no return- or push-based dominant with > 1 transmissible site at any snapshot | **not met as written**: the open pusher at L = 20 holds 2–3 sites (known from the Stage G scan) and the jump-based closer `… 21 4e 10 e5 …` (DJNZ) holds 2; restricted to pointer-closed non-block designs it holds at L = 16 (0 violations) and fails at L = 20 on the jump closer (seeds 6005, 6008) and on one LDI-based genome (6004) |
| Kill: L1 fails at both lengths with ≤ 1 site in ≥ 8/10 final dominants at both | **fired**: 9/10 and 8/10 |

## What the worlds did

- Every world closed (20/20): at L = 16 by 10,000–75,000 steps, at L = 20 by 20,000–500,000. The first replicator was a
  load–push word in 20/20 (`01 c5` 16, `21 e5` 3, `11 d5` 1).
- At ten million steps the most common tape is a minimal closer with no transmissible site in 17/20 worlds: the
  eight-byte return design `ad e3 21 e3 21 c0 ad c0` in 9/10 worlds at L = 16 (share 0.40–0.64 of the soup, 6 bits of
  capacity), the four-byte block-copy tiling `1e xx ed b0` in 8/10 at L = 20 (share 0.18–0.26; 8–10 bits).
- Genomes arise and are lost. In 3/20 worlds an irregular block copier carrying 10–12 transmissible sites, 7–10 of them
  never executed, was the most common tape for one to three million steps, always at a share below 0.3% (a diverse
  cloud, not a clone): seed 6006 (L = 16; `1e 50 c3 … ed b0`, 10 sites, 9 unexecuted, 1M–3M), seed 6003 (L = 20; LDDR
  genome, 11 sites, 10 unexecuted, 100k–300k) and seed 6004 (L = 20; an LDI-based genome `… 11 6c ed a0 c1 cf …`, 12
  sites, 7 unexecuted, 500k–1M, then an LDIR genome with 5 sites to 3M). Each was replaced: 6006 by an 8-byte LDIR
  tiling with 4 sites (share 0.6% at 10M), 6003 by an LDDR tiling with 3 executed sites, 6004 by the four-byte tiling
  with none (share 0.18 at 10M).
- Heritable fraction of random cells at the final snapshot (ten million steps; `final_func_rnd` in `assays.csv`, 16 cells per world): median 0.89 (L = 16), 0.95 (L = 20).

## Reading

The capacity for inherited variation does not recover after closure within ten million steps. The first individuals stay
canalised for as long as we watched; the genomes that reopen the channel, bytes copied as data and never run, appear in
3/20 worlds as rare variants and lose to smaller closers that carry nothing. Together with the Stage I observation
(lethal tar: genome-bearing block copiers at 5,000–50,000 steps replaced by the four-byte tiling) the direction is the
same in both regimes: selection in these soups favours the smallest closed design over the one that can vary. For the
paper: the genotype section shrinks to one paragraph stating this, with the three transient genomes as the evidence that
the channel exists and is not retained.
